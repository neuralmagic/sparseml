import time
import math
import torch
import logging
import numpy as np
import torch.nn as nn


from torch import Tensor
from torch.nn import Module


__all__ = [
    "safe_cholesky_inv",
    "AdaOBCHandle",
    "FisherOBCHandle"
]


_LOGGER = logging.getLogger(__name__)


def safe_cholesky_inv(X: Tensor, rel_damp: float = 1e-2):
    try:
        return torch.cholesky_inverse(torch.linalg.cholesky(X))
    except RuntimeError:
        reg = (rel_damp * torch.diag(X).mean()) * torch.eye(X.shape[0], device=X.device)
        return torch.cholesky_inverse(torch.linalg.cholesky(X + reg))


class AdaOBCHandle:

    def __init__(
        self, 
        layer: Module,
        dim_batch_size: int,
        num_samples: int,
        rel_damp: float = 0.0,
        verbose: bool = False
    ) -> None:
        assert isinstance(layer, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d))
        self.layer = layer
        self.num_samples = num_samples
        self.dim_batch_size = dim_batch_size
        self.rel_damp = rel_damp
        self.verbose = verbose
        # get weight
        W = layer.weight
        self.device = W.device
        # convert weight to the matrix form (d_out, d_in)
        self.dim_out = W.shape[0]
        self.dim_in  = np.prod(W.shape[1:])
        # init hessian
        self.H = None
        # init weight handle
        self.W = W
        # init the loss evolution
        self.losses = None
        # init weight traces
        self.traces = None


    def update_H(self, inp: Tensor) -> None:
        # allocate memory (if not initialized)
        if self.H is None:
            self.H = torch.zeros((self.dim_in, self.dim_in), device=self.device)
        # unfold inp if needed
        if isinstance(self.layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)
        else:
            inp = inp.view(-1, inp.shape[-1])
        self.H += (2 / self.num_samples) * inp.T @ inp


    def prepare(self) -> None:
        self.W = self.layer.weight.data.clone()
        if isinstance(self.layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            self.W = self.W.flatten(1)
        # if the entire input is 0 -> channel is dead and doesn't contribute
        dead = torch.diag(self.H) == 0
        self.H[dead, dead] = 1
        self.W[:, dead] = 0
        # prepare losses
        self.losses = torch.zeros((self.dim_out, self.dim_in + 1), device=self.device)
        # prepare traces
        self.traces = torch.zeros((self.dim_in + 1, self.dim_out, self.dim_in), device='cpu')


    def prepare_batch(self, i_start, i_end) -> tuple[Tensor]:
        W_batch = self.W[i_start:i_end, :]
        mask_batch = torch.zeros_like(W_batch).bool()
        return W_batch, mask_batch


    def prepare_batch_sparse(self, W_batch, mask_batch): 
        min_zeros = torch.sum((W_batch == 0), dim=1).min().item()
        # temporary hessian
        H_inv_batch = torch.empty((W_batch.shape[0], *self.H.shape), device=self.device)
        for i in range(W_batch.shape[0]):
            zero_ids = (W_batch[i] == 0)
            H_cur = self.H.clone()
            H_cur[zero_ids, :] = 0
            H_cur[:, zero_ids] = 0
            H_cur[zero_ids, zero_ids] = 1
            # invert
            H_inv_batch[i] = safe_cholesky_inv(H_cur)
            mask_batch[i, torch.nonzero(zero_ids, as_tuple=True)[0][:min_zeros]] = True

        return H_inv_batch, min_zeros


    def prepare_losses_and_traces(self) -> None:
        # prepare all
        self.prepare()

        _start = time.perf_counter()

        for i_start in range(0, self.dim_out, self.dim_batch_size):
            i_end = min(i_start + self.dim_batch_size, self.dim_out)
            batch_size = i_end - i_start
            batch_ids = torch.arange(batch_size, device=self.device)
            # prepare batch 
            W_batch, mask_batch = self.prepare_batch(i_start, i_end)
            H_inv_batch, min_nnz = self.prepare_batch_sparse(W_batch, mask_batch) 
            # init weight traces
            trace = torch.zeros((self.dim_in + 1, i_end - i_start, self.dim_in), device=self.device)
            trace[:(min_nnz + 1), :, :] = W_batch      

            for zeros in range(min_nnz + 1, self.dim_in + 1):
                _diag = torch.diagonal(H_inv_batch, dim1=1, dim2=2)
                scores = (W_batch ** 2) / _diag
                scores[mask_batch] = float('inf')
                pruned_id = torch.argmin(scores, 1)
                self.losses[i_start: i_end, zeros] = scores[batch_ids, pruned_id]
                row = H_inv_batch[batch_ids, pruned_id, :]
                d = _diag[batch_ids, pruned_id]
                W_batch -= row * (W_batch[batch_ids, pruned_id] / d).unsqueeze(1)
                mask_batch[batch_ids, pruned_id] = True
                W_batch[mask_batch] = 0
                trace[zeros, :, :] = W_batch
                # do not update on the last iteration
                if zeros == self.dim_in:
                    break
                row /= torch.sqrt(d).unsqueeze(1)
                H_inv_batch -= torch.bmm(row.unsqueeze(2), row.unsqueeze(1))

            self.losses[i_start: i_end, :] /= 2
            self.traces[:, i_start: i_end, :] = trace.cpu()

            torch.cuda.synchronize()

        _end = time.perf_counter()

        if self.verbose:
            _LOGGER.info(f'Preparation of losses and traces took {(_end - _start):.2f} s')


    def get_pruning_database(self, sparsities: np.ndarray) -> Tensor:
        losses = self.losses[:, 1:].reshape(-1)
        order = torch.argsort(losses)
        Ws = torch.zeros((len(sparsities), self.dim_out, self.dim_in), device=self.device)
        cum_losses = [0] * len(sparsities)

        for i in range(self.dim_out):
            for j, sparsity in enumerate(sparsities):
                count = int(math.ceil(self.dim_out * self.dim_in * sparsity))
                perrow = torch.sum(
                    torch.div(order[:count], self.dim_in, rounding_mode='trunc') == i
                ).item()
                cum_losses[j] += torch.sum(self.losses[i, :(perrow + 1)]).item()
                Ws[j, i, :] = self.traces[perrow, i, :].to(self.device)
        
        if self.verbose:
            for sparsity, cum_loss in zip(sparsities, cum_losses):
                _LOGGER.info(f'Sparsity: {sparsity:.3f} / Loss: {cum_loss:.4f}')

        # free memory
        self.free()

        return Ws


    def free(self) -> None:
        self.H = None
        self.W = None
        self.losses = None
        self.traces = None
        torch.cuda.empty_cache()


class FisherOBCHandle:

    def __init__(
        self, 
        weight: Tensor,
        obc_batch_size: int,
        verbose: bool = False,
    ) -> None:
        self.verbose = verbose
        self.device = weight.device
        self.obc_batch_size = obc_batch_size
        # convert weight to the matrix form (d_out, d_in)
        self.dim_out = weight.shape[0]
        self.dim_in  = np.prod(weight.shape[1:])
        # init Finv
        self.Finv = None
        # init weight handle
        self.W = weight
        # init the loss evolution
        self.losses = None
        # init weight traces
        self.traces = None

    
    def set_Finv(self, Finv: Tensor):
        assert Finv.shape == (self.dim_out, self.dim_in, self.dim_in), \
            "Expected block Fisher inverse shape (dim_out, dim_in, dim_in)"
        self.Finv = Finv.to(self.device)


    def prepare(self) -> None:
        # prepare losses
        self.losses = torch.zeros((self.dim_out, self.dim_in), device=self.device)
        # prepare traces
        self.traces = torch.zeros((self.dim_in + 1, self.dim_out, self.dim_in), device='cpu')


    def prepare_batch(self, i_start, i_end) -> tuple:
        W_batch = self.W[i_start: i_end, :]
        M_batch = torch.zeros_like(W_batch).bool()
        Hinv_batch = self.Finv[i_start: i_end, :]
        # get minimum number of zeros in a row
        min_zeros = torch.sum((W_batch == 0), dim=1).min().item()
        for i in range(W_batch.shape[0]):
            zero_ids = (W_batch[i] == 0)
            M_batch[i, torch.nonzero(zero_ids, as_tuple=True)[0][:min_zeros]] = True
        return W_batch, M_batch, Hinv_batch, min_zeros


    def prepare_losses_and_traces(self) -> None:
        # prepare all
        self.prepare()

        _start = time.perf_counter()

        for i_start in range(0, self.dim_out, self.obc_batch_size):
            i_end = min(i_start + self.obc_batch_size, self.dim_out)
            # get current batch size
            batch_size = i_end - i_start
            batch_ids = torch.arange(batch_size, device=self.device)
            shifted_batch_ids = torch.arange(i_start, i_end, device=self.device)
            # prepare batch 
            W_batch, M_batch, H_inv_batch, min_nnz = self.prepare_batch(i_start, i_end)
            # init weight traces
            trace = torch.zeros((self.dim_in + 1, i_end - i_start, self.dim_in), device=self.device)
            trace[:(min_nnz + 1), :, :] = W_batch
            # get list of current losses
            cur_losses = torch.zeros(batch_size, device=self.device)  
            for zeros in range(min_nnz + 1, self.dim_in + 1):
                H_inv_batch_diag = torch.diagonal(H_inv_batch, dim1=1, dim2=2)
                scores = (W_batch ** 2) / H_inv_batch_diag
                scores[M_batch] = float('inf')
                min_scores, pruned_id = torch.min(scores, 1)
                cur_losses += min_scores
                self.losses[shifted_batch_ids, pruned_id] = cur_losses
                row = H_inv_batch[batch_ids, pruned_id, :]
                d = H_inv_batch_diag[batch_ids, pruned_id]
                W_batch -= row * (W_batch[batch_ids, pruned_id] / d).unsqueeze(1)
                M_batch[batch_ids, pruned_id] = True
                W_batch[M_batch] = 0
                trace[zeros, :, :] = W_batch
                # do not update on the last iteration
                if zeros == self.dim_in:
                    break
                row /= torch.sqrt(d).unsqueeze(1)
                H_inv_batch -= torch.bmm(row.unsqueeze(2), row.unsqueeze(1))

            self.losses[i_start: i_end, :] /= 2
            self.traces[:, i_start: i_end, :] = trace.cpu()

            torch.cuda.synchronize()

        _end = time.perf_counter()

        if self.verbose:
            _LOGGER.info(f'Preparation of losses and traces took {(_end - _start):.2f} s')

            

    def get_pruning_database(self, sparsities: np.ndarray) -> list[Tensor]:
        sorted_losses, _ = torch.sort(self.losses.view(-1))
        # prepare list of weight for every sparsity level of interest
        Ws = [torch.zeros((self.dim_out, self.dim_in), device=self.device) for _ in sparsities]
        # 
        for i, sparsity in enumerate(sparsities):
            num_zeros = int(math.ceil(self.dim_out * self.dim_in * sparsity))
            # loss threshold
            loss_thr = sorted_losses[num_zeros]
            for row in range(self.dim_out):
                num_zeros_in_row = torch.count_nonzero(self.losses[row, :] <= loss_thr)
                Ws[i][row, :] = self.traces[num_zeros_in_row, row, :]

        # free memory
        self.free()

        return Ws

    def free(self) -> None:
        self.Finv = None
        self.W = None
        self.losses = None
        self.traces = None
        torch.cuda.empty_cache()
