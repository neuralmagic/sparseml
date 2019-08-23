import os
from tqdm import tqdm
import math
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.nn import Linear
from torch.nn.modules.conv import _ConvNd
from tensorboardX import SummaryWriter

from neuralmagicML.models import save_model
from neuralmagicML.models import resnet50
from neuralmagicML.datasets import ImagenetteDataset, ImageNetDataset, EarlyStopDataset
from neuralmagicML.sparsity import (
    LearningRateModifier, ASRegModifier, ScheduledModifierManager, ScheduledOptimizer,
    ASAnalyzerLayer, ASAnalyzerModule
)
from neuralmagicML.utils import CrossEntropyLossWrapper, TopKAccuracy


def reg_fine_tuning(reg_tens, reg_func, alpha, lr_init, lr_final, lr_updates, epochs,
                    model_pretrained, model_id, device,
                    dataset_root, num_classes, train_batch_size, test_batch_size):
    train_dataset = ImagenetteDataset(dataset_root, train=True, rand_trans=True)
    train_data_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=4)
    print('train dataset created: \n{}\n'.format(train_dataset))

    val_dataset = ImagenetteDataset(dataset_root, train=False, rand_trans=False)
    val_data_loader = DataLoader(val_dataset, batch_size=test_batch_size, shuffle=False, num_workers=4)
    print('validation test dataset created: \n{}\n'.format(val_dataset))

    train_test_dataset = EarlyStopDataset(
        ImagenetteDataset(dataset_root, train=True, rand_trans=False),
                          early_stop=len(val_dataset) if len(val_dataset) > 1000 else round(0.1 * len(train_dataset)))
    train_test_data_loader = DataLoader(train_test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=4)
    print('train test dataset created: \n{}\n'.format(train_test_dataset))

    model = resnet50(pretrained=model_pretrained, num_classes=num_classes)
    model = model.to(device)

    lr_update_freq = epochs / (lr_updates + 1.0)
    lr_gamma = (lr_final / lr_init) ** (1 / lr_updates)

    ### optimizer definitions
    momentum = 0.9
    weight_decay = 1e-4
    ###

    lr_modifier = LearningRateModifier(lr_class='ExponentialLR', lr_kwargs={'gamma': lr_gamma},
                                       start_epoch=0.0, end_epoch=epochs,
                                       update_frequency=lr_update_freq)
    modify_layers = [name for name, mod in model.named_modules() if isinstance(mod, _ConvNd)]

    # remove the first conv if we are working on the input to each conv
    if reg_tens == 'inp':
        modify_layers = modify_layers[1:]
    # remove the last conv if we are working on the output from each conv
    elif reg_tens == 'out':
        modify_layers = modify_layers[:-1]

    as_reg_modifier = ASRegModifier(modify_layers, alpha, reg_func, reg_tens, start_epoch=0.0)

    modifier_manager = ScheduledModifierManager([lr_modifier, as_reg_modifier])
    print('\nCreated ScheduledModifierManager with exponential lr_modifier with gamma {} and AS reg modifier'
          .format(lr_gamma))

    optimizer = optim.SGD(model.parameters(), lr_init, momentum=momentum,
                          weight_decay=weight_decay, nesterov=True)
    optimizer = ScheduledOptimizer(optimizer, model, modifier_manager, steps_per_epoch=len(train_dataset))
    print('\nCreated scheudled optimizer with initial lr: {}, momentum: {}, weight decay: {}'
          .format(lr_init, momentum, weight_decay))

    loss = CrossEntropyLossWrapper(extras={'top1acc': TopKAccuracy(1)})
    print('\nCreated loss wrapper\n{}'.format(loss))

    logs_dir = os.path.abspath(os.path.expanduser(os.path.join('.', 'model_training_logs', model_id)))

    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    writer = SummaryWriter(logdir=logs_dir, comment='imagenette training')
    print('\nCreated summary writer logging to \n{}'.format(logs_dir))

    def test_epoch(model, data_loader, loss, device, epoch):
        model.eval()
        results = {}

        with torch.no_grad():
            for batch, (*x_feature, y_lab) in enumerate(tqdm(data_loader)):
                y_lab = y_lab.to(device)
                x_feature = tuple([dat.to(device) for dat in x_feature])
                batch_size = y_lab.shape[0]
                y_pred = model(*x_feature)
                losses = loss(x_feature, y_lab, y_pred)

                for key, val in losses.items():
                    if key not in results:
                        results[key] = []
                    result = val.detach_().cpu()
                    result = result.repeat(batch_size)
                    results[key].append(result)

        return results

    def test_epoch_writer(model, data_loader, loss, device, epoch, writer, key):
        losses = test_epoch(model, data_loader, loss, device, epoch)

        for loss, values in losses.items():
            val = torch.mean(torch.cat(values))
            writer.add_scalar(key.format(loss), val, epoch)
            print('{}: {}'.format(loss, val))

    def test_as_values(as_model, data_loader, device, epoch, writer, sample_size=1000):
        as_model.eval()
        as_model.clear_layers()
        as_model.enable_layers()
        sample_count = 0

        with torch.no_grad():
            for batch, (*x_feature, y_lab) in enumerate(tqdm(data_loader)):
                y_lab = y_lab.to(device)
                x_feature = tuple([dat.to(device) for dat in x_feature])
                batch_size = y_lab.shape[0]
                y_pred = model(*x_feature)
                sample_count += batch_size

                if sample_count >= sample_size:
                    break

        as_model.disable_layers()

        for name, layer in as_model.layers.items():
            writer.add_scalar('Act Sparsity/{}'.format(name), layer.inputs_sparsity_mean, epoch)

        as_model.clear_layers()

    def train_epoch(model, data_loader, optimizer, loss, device, epoch, writer):
        model.train()
        init_batch_size = None
        batches_per_epoch = len(data_loader)

        for batch, (*x_feature, y_lab) in enumerate(tqdm(data_loader)):
            y_lab = y_lab.to(device)
            x_feature = tuple([dat.to(device) for dat in x_feature])
            batch_size = y_lab.shape[0]
            if init_batch_size is None:
                init_batch_size = batch_size
            optimizer.zero_grad()
            y_pred = model(*x_feature)
            losses = loss(x_feature, y_lab, y_pred)
            losses['loss'] = optimizer.loss_update(losses['loss'])  # update loss with the AS modifier regularization
            losses['loss'].backward()
            optimizer.step(closure=None)

            step_count = init_batch_size * (epoch * batches_per_epoch + batch)
            for _loss, _value in losses.items():
                writer.add_scalar('Train/{}'.format(_loss), _value.item(), step_count)
                writer.add_scalar('Train/Learning Rate', optimizer.learning_rate, step_count)


    print('Training model...')

    analyzer_model = ASAnalyzerModule(
        model, [ASAnalyzerLayer(name, division=0, track_inputs_sparsity=True)
                for name, mod in model.named_modules() if isinstance(mod, _ConvNd) or isinstance(mod, Linear)]
    )
    print('\nCreated AS analyzer module')

    print('Running initial validation values for later comparison')
    # test_epoch_writer(model, val_data_loader, loss, device, -1, writer, 'Test/Validation/{}')
    # test_as_values(analyzer_model, val_data_loader, device, -1, writer)

    for epoch in tqdm(range(math.ceil(modifier_manager.max_epochs))):
        print('Starting epoch {}'.format(epoch))
        optimizer.epoch_start()
        train_epoch(model, train_data_loader, optimizer, loss, device, epoch, writer)

        print('Completed training for epoch {}, testing validation dataset'.format(epoch))
        test_epoch_writer(model, val_data_loader, loss, device, epoch, writer, 'Test/Validation/{}')

        print('Completed testing validation dataset for epoch {}, testing training dataset'.format(epoch))
        test_epoch_writer(model, train_test_data_loader, loss, device, epoch, writer, 'Test/Training/{}')

        print('Completed testing validation dataset for epoch {}, testing activation sparsity'.format(epoch))
        test_as_values(analyzer_model, val_data_loader, optimizer, loss, device, epoch, writer)

        optimizer.epoch_end()

    save_path = os.path.abspath(os.path.expanduser(os.path.join('.', '{}.pth'.format(model_id))))
    print('Finished training, saving model to {}'.format(save_path))
    save_model(save_path, model, optimizer, epoch)
    print('Saved model')


if __name__ == '__main__':
    reg_fine_tuning('inp', 'l1', 0.00001, 0.01, 0.0001, 5, 30,
                    'imagenette/dense', 'resnet-test', 'cpu',
                    os.path.abspath(os.path.expanduser('~/datasets/imagenette')), 10, 1, 1)
