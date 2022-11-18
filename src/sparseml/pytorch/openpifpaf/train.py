# Adapted from https://github.com/openpifpaf/openpifpaf

# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import copy
import logging
import os
import socket

import torch

import openpifpaf
from openpifpaf import __version__
from openpifpaf.train import default_output_file
from sparseml.pytorch.openpifpaf.utils import SparseMLTrainer


LOG = logging.getLogger(__name__)


def cli():
    parser = argparse.ArgumentParser(
        prog="python3 -m openpifpaf.train",
        usage="%(prog)s [options]",
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version="OpenPifPaf {version}".format(version=__version__),
    )
    parser.add_argument("-o", "--output", default=None, help="output file")
    parser.add_argument("--disable-cuda", action="store_true", help="disable CUDA")
    parser.add_argument(
        "--ddp",
        default=False,
        action="store_true",
        help="[experimental] DistributedDataParallel",
    )
    default_local_rank = os.environ.get("LOCAL_RANK")
    if default_local_rank is not None:
        default_local_rank = int(default_local_rank)
    parser.add_argument(
        "--local_rank",
        default=default_local_rank,
        type=int,
        help="[experimental] for torch.distributed.launch",
    )
    parser.add_argument(
        "--no-sync-batchnorm",
        dest="sync_batchnorm",
        default=True,
        action="store_false",
        help="[experimental] in ddp, to not use syncbatchnorm",
    )

    openpifpaf.logger.cli(parser)
    openpifpaf.network.Factory.cli(parser)
    openpifpaf.network.losses.Factory.cli(parser)
    openpifpaf.network.Trainer.cli(parser)
    openpifpaf.encoder.cli(parser)
    openpifpaf.optimize.cli(parser)
    openpifpaf.datasets.cli(parser)
    openpifpaf.show.cli(parser)
    openpifpaf.visualizer.cli(parser)

    args = parser.parse_args()

    openpifpaf.logger.configure(args, LOG)
    if args.log_stats:
        logging.getLogger("openpifpaf.stats").setLevel(logging.DEBUG)

    # DDP with SLURM
    slurm_process_id = os.environ.get("SLURM_PROCID")
    if args.ddp and slurm_process_id is not None:
        if torch.cuda.device_count() > 1:
            LOG.warning(
                "Expected one GPU per SLURM task but found %d. "
                'Try with "srun --gpu-bind=closest ...". Still trying.',
                torch.cuda.device_count(),
            )

        # if there is more than one GPU available, assume that other SLURM tasks
        # have access to the same GPUs and assign GPUs uniquely by slurm_process_id
        args.local_rank = (
            int(slurm_process_id) % torch.cuda.device_count()
            if torch.cuda.device_count() > 0
            else 0
        )

        os.environ["RANK"] = slurm_process_id
        if not os.environ.get("WORLD_SIZE") and os.environ.get("SLURM_NTASKS"):
            os.environ["WORLD_SIZE"] = os.environ.get("SLURM_NTASKS")

        LOG.info("found SLURM process id: %s", slurm_process_id)
        LOG.info(
            "distributed env: master=%s port=%s rank=%s world=%s, "
            "local rank (GPU)=%d",
            os.environ.get("MASTER_ADDR"),
            os.environ.get("MASTER_PORT"),
            os.environ.get("RANK"),
            os.environ.get("WORLD_SIZE"),
            args.local_rank,
        )

    # add args.device
    args.device = torch.device("cpu")
    args.pin_memory = False
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device("cuda")
        args.pin_memory = True
    LOG.info(
        "neural network device: %s (CUDA available: %s, count: %d)",
        args.device,
        torch.cuda.is_available(),
        torch.cuda.device_count(),
    )

    # output
    if args.output is None:
        args.output = default_output_file(args)
        os.makedirs("outputs", exist_ok=True)

    openpifpaf.network.Factory.configure(args)
    openpifpaf.network.losses.Factory.configure(args)
    openpifpaf.network.Trainer.configure(args)
    openpifpaf.encoder.configure(args)
    openpifpaf.datasets.configure(args)
    openpifpaf.show.configure(args)
    openpifpaf.visualizer.configure(args)

    return args


def main():
    args = cli()

    datamodule = openpifpaf.datasets.factory(args.dataset)

    net_cpu, start_epoch = openpifpaf.network.Factory().factory(
        head_metas=datamodule.head_metas
    )
    loss = openpifpaf.network.losses.Factory().factory(datamodule.head_metas)

    checkpoint_shell = None
    if not args.disable_cuda and torch.cuda.device_count() > 1 and not args.ddp:
        LOG.info("Multiple GPUs with DataParallel: %d", torch.cuda.device_count())
        checkpoint_shell = copy.deepcopy(net_cpu)
        net = torch.nn.DataParallel(net_cpu.to(device=args.device))
        loss = loss.to(device=args.device)
    elif not args.disable_cuda and torch.cuda.device_count() == 1 and not args.ddp:
        LOG.info("Single GPU training")
        checkpoint_shell = copy.deepcopy(net_cpu)
        net = net_cpu.to(device=args.device)
        loss = loss.to(device=args.device)
    elif not args.disable_cuda and torch.cuda.device_count() > 0:
        LOG.info("Multiple GPUs with DistributedDataParallel")
        assert not list(loss.parameters())
        assert torch.cuda.device_count() > 0
        checkpoint_shell = copy.deepcopy(net_cpu)
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        LOG.info(
            "DDP: rank %d, world %d",
            torch.distributed.get_rank(),
            torch.distributed.get_world_size(),
        )
        if args.sync_batchnorm:
            LOG.info("convert all batchnorms to syncbatchnorms")
            net_cpu = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net_cpu)
        else:
            LOG.info("not converting batchnorms to syncbatchnorms")
        net = torch.nn.parallel.DistributedDataParallel(
            net_cpu.to(device=args.device),
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=isinstance(
                datamodule, openpifpaf.datasets.MultiDataModule
            ),
        )
        loss = loss.to(device=args.device)
    else:
        net = net_cpu

    openpifpaf.logger.train_configure(args)
    train_loader = datamodule.train_loader()
    val_loader = datamodule.val_loader()
    if torch.distributed.is_initialized():
        train_loader = datamodule.distributed_sampler(train_loader)
        val_loader = datamodule.distributed_sampler(val_loader)

    optimizer = openpifpaf.optimize.factory_optimizer(
        args, list(net.parameters()) + list(loss.parameters())
    )
    lr_scheduler = openpifpaf.optimize.factory_lrscheduler(
        args, optimizer, len(train_loader), last_epoch=start_epoch
    )
    trainer = SparseMLTrainer(
        net,
        loss,
        optimizer,
        args.output,
        checkpoint_shell=checkpoint_shell,
        lr_scheduler=lr_scheduler,
        device=args.device,
        model_meta_data={
            "args": vars(args),
            "version": __version__,
            "plugin_versions": openpifpaf.plugin.versions(),
            "hostname": socket.gethostname(),
        },
    )
    trainer.loop(train_loader, val_loader, start_epoch=start_epoch)


if __name__ == "__main__":
    main()
