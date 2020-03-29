from typing import List
from torch import Tensor
from torch.nn import (
    Module,
    Conv2d,
    BatchNorm2d,
    MaxPool2d,
    Linear,
    init,
    Sequential,
    Dropout,
    Softmax,
    Sigmoid,
)

from neuralmagicML.pytorch.nn import ReLU
from neuralmagicML.pytorch.utils.model import load_pretrained_model, MODEL_MAPPINGS

__all__ = [
    "VGG",
    "vgg11",
    "vgg11_bn",
    "vgg13",
    "vgg13_bn",
    "vgg16",
    "vgg16_bn",
    "vgg19_bn",
    "vgg19",
]


def _init_conv(conv: Conv2d):
    init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")

    if conv.bias is not None:
        init.constant_(conv.bias, 0)


def _init_batch_norm(norm: BatchNorm2d):
    init.constant_(norm.weight, 1.0)
    init.constant_(norm.bias, 0.0)


def _init_linear(linear: Linear):
    init.normal_(linear.weight, 0, 0.01)
    init.constant_(linear.bias, 0)


class _Block(Module):
    def __init__(self, in_channels: int, out_channels: int, batch_norm: bool):
        super().__init__()
        self.conv = Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, stride=1
        )
        self.bn = BatchNorm2d(out_channels) if batch_norm else None
        self.act = ReLU(num_channels=out_channels, inplace=True)

        self.initialize()

    def forward(self, inp: Tensor):
        out = self.conv(inp)

        if self.bn is not None:
            out = self.bn(out)

        out = self.act(out)

        return out

    def initialize(self):
        _init_conv(self.conv)

        if self.bn is not None:
            _init_batch_norm(self.bn)


class _Classifier(Module):
    def __init__(self, in_channels: int, num_classes: int, class_type: str = "single"):
        super().__init__()
        self.mlp = Sequential(
            Linear(in_channels * 7 * 7, 4096),
            Dropout(),
            ReLU(num_channels=4096, inplace=True),
            Linear(4096, 4096),
            Dropout(),
            ReLU(num_channels=4096, inplace=True),
            Linear(4096, num_classes),
        )

        if class_type == "single":
            self.softmax = Softmax(dim=1)
        elif class_type == "multi":
            self.softmax = Sigmoid()
        else:
            raise ValueError("unknown class_type given of {}".format(class_type))

    def forward(self, inp: Tensor):
        out = inp.view(inp.size(0), -1)
        logits = self.mlp(out)
        classes = self.softmax(logits)

        return logits, classes


class VGGSectionSettings(object):
    def __init__(
        self, num_blocks: int, in_channels: int, out_channels: int, use_batchnorm: bool
    ):
        """
        :param num_blocks: the number of blocks to put in the section (conv [bn] relu)
        :param in_channels: the number of input channels to the section
        :param out_channels: the number of output channels from the section
        :param use_batchnorm: True to put batchnorm after each conv, False otherwise
        """
        self.num_blocks = num_blocks
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_batchnorm = use_batchnorm


class VGG(Module):
    def __init__(
        self,
        sec_settings: List[VGGSectionSettings],
        model_arch_tag: str,
        num_classes: int = 1000,
        class_type: str = "single",
        pretrained: bool = False,
    ):
        """
        Standard VGG model
        https://arxiv.org/abs/1409.1556

        :param sec_settings: the settings for each section in the vgg model
        :param model_arch_tag: the architecture tag used for loading pretrained weights: ex vgg/16, vgg/16bn
        :param num_classes: the number of classes to classify
        :param pretrained: True to load dense, pretrained weights from imagenet, false otherwise
                           Additionally can specify other available datsets (dataset/dense) and
                           kernel sparsity models (dataset/sparse, dataset/sparse-perf)
        """
        super(VGG, self).__init__()
        self.sections = Sequential(
            *[VGG.create_section(settings) for settings in sec_settings]
        )
        self.classifier = _Classifier(
            sec_settings[-1].out_channels, num_classes, class_type
        )

        if pretrained:
            pretrained_key = pretrained if isinstance(pretrained, str) else ""
            load_pretrained_model(
                self,
                pretrained_key,
                model_arch=model_arch_tag,
                ignore_tensors=None
                if num_classes == 1000
                else ["classifier.mlp.6.weight", "classifier.mlp.6.bias"],
            )

    def forward(self, inp):
        out = self.sections(inp)
        logits, classes = self.classifier(out)

        return logits, classes

    @staticmethod
    def create_section(settings: VGGSectionSettings) -> Sequential:
        blocks = []
        in_channels = settings.in_channels

        for _ in range(settings.num_blocks):
            blocks.append(
                _Block(in_channels, settings.out_channels, settings.use_batchnorm)
            )
            in_channels = settings.out_channels

        blocks.append(MaxPool2d(kernel_size=2, stride=2))

        return Sequential(*blocks)


def vgg11(**kwargs) -> VGG:
    sec_settings = [
        VGGSectionSettings(
            num_blocks=1, in_channels=3, out_channels=64, use_batchnorm=False
        ),
        VGGSectionSettings(
            num_blocks=1, in_channels=64, out_channels=128, use_batchnorm=False
        ),
        VGGSectionSettings(
            num_blocks=2, in_channels=128, out_channels=256, use_batchnorm=False
        ),
        VGGSectionSettings(
            num_blocks=2, in_channels=256, out_channels=512, use_batchnorm=False
        ),
        VGGSectionSettings(
            num_blocks=2, in_channels=512, out_channels=512, use_batchnorm=False
        ),
    ]

    return VGG(sec_settings=sec_settings, model_arch_tag="vgg/11", **kwargs)


MODEL_MAPPINGS["vgg11"] = vgg11


def vgg11_bn(**kwargs) -> VGG:
    sec_settings = [
        VGGSectionSettings(
            num_blocks=1, in_channels=3, out_channels=64, use_batchnorm=True
        ),
        VGGSectionSettings(
            num_blocks=1, in_channels=64, out_channels=128, use_batchnorm=True
        ),
        VGGSectionSettings(
            num_blocks=2, in_channels=128, out_channels=256, use_batchnorm=True
        ),
        VGGSectionSettings(
            num_blocks=2, in_channels=256, out_channels=512, use_batchnorm=True
        ),
        VGGSectionSettings(
            num_blocks=2, in_channels=512, out_channels=512, use_batchnorm=True
        ),
    ]

    return VGG(sec_settings=sec_settings, model_arch_tag="vgg/11-bn", **kwargs)


MODEL_MAPPINGS["vgg11_bn"] = vgg11_bn


def vgg13(**kwargs) -> VGG:
    sec_settings = [
        VGGSectionSettings(
            num_blocks=2, in_channels=3, out_channels=64, use_batchnorm=False
        ),
        VGGSectionSettings(
            num_blocks=2, in_channels=64, out_channels=128, use_batchnorm=False
        ),
        VGGSectionSettings(
            num_blocks=2, in_channels=128, out_channels=256, use_batchnorm=False
        ),
        VGGSectionSettings(
            num_blocks=2, in_channels=256, out_channels=512, use_batchnorm=False
        ),
        VGGSectionSettings(
            num_blocks=2, in_channels=512, out_channels=512, use_batchnorm=False
        ),
    ]

    return VGG(sec_settings=sec_settings, model_arch_tag="vgg/13", **kwargs)


MODEL_MAPPINGS["vgg13"] = vgg13


def vgg13_bn(**kwargs) -> VGG:
    sec_settings = [
        VGGSectionSettings(
            num_blocks=2, in_channels=3, out_channels=64, use_batchnorm=True
        ),
        VGGSectionSettings(
            num_blocks=2, in_channels=64, out_channels=128, use_batchnorm=True
        ),
        VGGSectionSettings(
            num_blocks=2, in_channels=128, out_channels=256, use_batchnorm=True
        ),
        VGGSectionSettings(
            num_blocks=2, in_channels=256, out_channels=512, use_batchnorm=True
        ),
        VGGSectionSettings(
            num_blocks=2, in_channels=512, out_channels=512, use_batchnorm=True
        ),
    ]

    return VGG(sec_settings=sec_settings, model_arch_tag="vgg/13-bn", **kwargs)


MODEL_MAPPINGS["vgg13_bn"] = vgg13_bn


def vgg16(**kwargs) -> VGG:
    sec_settings = [
        VGGSectionSettings(
            num_blocks=2, in_channels=3, out_channels=64, use_batchnorm=False
        ),
        VGGSectionSettings(
            num_blocks=2, in_channels=64, out_channels=128, use_batchnorm=False
        ),
        VGGSectionSettings(
            num_blocks=3, in_channels=128, out_channels=256, use_batchnorm=False
        ),
        VGGSectionSettings(
            num_blocks=3, in_channels=256, out_channels=512, use_batchnorm=False
        ),
        VGGSectionSettings(
            num_blocks=3, in_channels=512, out_channels=512, use_batchnorm=False
        ),
    ]

    return VGG(sec_settings=sec_settings, model_arch_tag="vgg/16", **kwargs)


MODEL_MAPPINGS["vgg16"] = vgg16


def vgg16_bn(**kwargs) -> VGG:
    sec_settings = [
        VGGSectionSettings(
            num_blocks=2, in_channels=3, out_channels=64, use_batchnorm=True
        ),
        VGGSectionSettings(
            num_blocks=2, in_channels=64, out_channels=128, use_batchnorm=True
        ),
        VGGSectionSettings(
            num_blocks=3, in_channels=128, out_channels=256, use_batchnorm=True
        ),
        VGGSectionSettings(
            num_blocks=3, in_channels=256, out_channels=512, use_batchnorm=True
        ),
        VGGSectionSettings(
            num_blocks=3, in_channels=512, out_channels=512, use_batchnorm=True
        ),
    ]

    return VGG(sec_settings=sec_settings, model_arch_tag="vgg/16-bn", **kwargs)


MODEL_MAPPINGS["vgg16_bn"] = vgg16_bn


def vgg19(**kwargs) -> VGG:
    sec_settings = [
        VGGSectionSettings(
            num_blocks=2, in_channels=3, out_channels=64, use_batchnorm=False
        ),
        VGGSectionSettings(
            num_blocks=2, in_channels=64, out_channels=128, use_batchnorm=False
        ),
        VGGSectionSettings(
            num_blocks=4, in_channels=128, out_channels=256, use_batchnorm=False
        ),
        VGGSectionSettings(
            num_blocks=4, in_channels=256, out_channels=512, use_batchnorm=False
        ),
        VGGSectionSettings(
            num_blocks=4, in_channels=512, out_channels=512, use_batchnorm=False
        ),
    ]

    return VGG(sec_settings=sec_settings, model_arch_tag="vgg/19", **kwargs)


MODEL_MAPPINGS["vgg19"] = vgg19


def vgg19_bn(**kwargs) -> VGG:
    sec_settings = [
        VGGSectionSettings(
            num_blocks=2, in_channels=3, out_channels=64, use_batchnorm=True
        ),
        VGGSectionSettings(
            num_blocks=2, in_channels=64, out_channels=128, use_batchnorm=True
        ),
        VGGSectionSettings(
            num_blocks=4, in_channels=128, out_channels=256, use_batchnorm=True
        ),
        VGGSectionSettings(
            num_blocks=4, in_channels=256, out_channels=512, use_batchnorm=True
        ),
        VGGSectionSettings(
            num_blocks=4, in_channels=512, out_channels=512, use_batchnorm=True
        ),
    ]

    return VGG(sec_settings=sec_settings, model_arch_tag="vgg/19-bn", **kwargs)


MODEL_MAPPINGS["vgg19_bn"] = vgg19_bn
