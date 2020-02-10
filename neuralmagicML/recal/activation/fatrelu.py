from abc import abstractmethod
from typing import Union, List, Dict
from enum import Enum
import torch
from torch import Tensor
from torch.nn import Module, Parameter, ReLU
import torch.nn.functional as TF


__all__ = [
    "fat_relu",
    "fat_pw_relu",
    "fat_sig_relu",
    "fat_exp_relu",
    "FATReLU",
    "FATPWReLU",
    "FATSigReLU",
    "FATExpReLU",
    "FATReluType",
    "convert_relus_to_fat",
    "set_relu_to_fat",
]


def _apply_permuted_channels(apply_fn, tens: Tensor, **kwargs):
    if len(tens.shape) < 3:
        return apply_fn(tens, **kwargs)

    perm = [ind for ind in range(len(tens.shape))]
    # swap the channel and the last element so we can broadcast across the channels
    perm[1] = perm[-1]
    perm[-1] = 1

    return apply_fn(tens.permute(perm), **kwargs).permute(perm)


def fat_relu(tens: Tensor, threshold: Union[Tensor, float], inplace: bool) -> Tensor:
    """
    :param tens: the tensor to apply the fat relu to
    :param threshold: the threshold to apply
                      if not a single value then the dimension to broadcast across must be last in the tensor
    :param inplace: false to create a new tensor, true to overwrite the current tensor's values
    :return: f(x, t) = 0 if x < t
             x if x >= t
    """
    if isinstance(threshold, float):
        # not channelwise, can get by with using a threshold
        return TF.threshold(tens, threshold, 0.0, inplace)

    mask = (tens >= threshold).float()
    out = tens * mask if not inplace else tens.mul_(mask)

    return out


def fat_pw_relu(
    tens: Tensor, threshold: Tensor, compression: Tensor, inplace: bool
) -> Tensor:
    """
    :param tens: the tensor to apply the piecewise fat relu to
    :param threshold: the threshold at which all values will be zero or interpolated between threshold and 0
    :param compression: the compression or slope to interpolate between 0 and the threshold with
    :param inplace: false to create a new tensor, true to overwrite the current tensor's values
    :return: f(x, t, c) = 0 if x <= (t - t/c)
                        = x if x >= t
                        = c(x - (t - t/c)) if x > (t - t/c) and x < t
    """
    x_offset = threshold - threshold / compression

    # apply the fat relu up until our x_offset (where our compression region starts)
    out = fat_relu(tens, x_offset, inplace)

    # calculate the compression region values
    comp_mask = ((tens < threshold) * tens > x_offset).float()
    comp_tens = compression * (out - x_offset)

    # reassign the compression values in the output
    out = (
        (-1.0 * comp_mask + 1.0) * out + comp_tens * comp_mask
        if not inplace
        else out.mul_(-1.0 * comp_mask + 1.0).add_(comp_tens * comp_mask)
    )

    return out


def fat_sig_relu(tens: Tensor, threshold: Tensor, compression: Tensor) -> Tensor:
    """
    no option for inplace with this function since it is non linear

    :param tens: the tensor to apply the sigmoid fat relu to
    :param threshold: the threshold at which all values will be zero or approximated in the sigmoid region
    :param compression: the compression or slope to use in the sigmoid region
    :return: f(x, t, c) = x / e^(c*(t-x))
    """
    out = tens / (1.0 + torch.exp(compression * (threshold - tens)))
    out = TF.relu(
        out, inplace=True
    )  # make sure that the negative region is always zero activation with a regular ReLU

    return out


def fat_exp_relu(tens: Tensor, threshold: Tensor, compression: Tensor) -> Tensor:
    """
    no option for inplace with this function since it is non linear

    :param tens: the tensor to apply the exponential fat relu to
    :param threshold: the threshold at which all values will be zero or approximated in the exponential region
    :param compression: the compression or slope to use in the exponential region
    :return: f(x, t, c) = 0 if x <= 0
                        = x if x >= t
                        = x * e^(c(x-t)) if x > 0 and x < t
    """
    # remove the negative values
    out = TF.relu(tens)
    # calculate the compression region values
    comp_mask = ((out < threshold) * (out > 0.0)).float()

    comp_tens = out * torch.exp(compression * (out - threshold))

    # reassign the compression values in the output
    out = (-1.0 * comp_mask + 1.0) * out + comp_tens * comp_mask

    return out


class FATReLU(Module):
    def __init__(
        self, threshold: Union[float, List[float]] = 0.0, inplace: bool = False
    ):
        """
        Applies a FAT ReLU (forced activation threshold) over the input.
        Instead of setting all negative values to 0 like with ReLU, this sets all values < threshold equal to 0

        :param threshold: the threshold that all values < threshold will be set to 0
                          if type float then f(x) = x if x >= threshold else 0
                          if type list then f(x[:, chan]) = x[:, chan] if x[:, chan] >= threshold[chan] else 0
                          if type list and empty, applies activation the list option but dynamically initialized to the num chan
        :param inplace: perform the operation inplace or create a new tensor
        """
        super(FATReLU, self).__init__()
        self._dynamic = False
        self._channel_wise = False
        self._num_channels = None

        if isinstance(threshold, List):
            self._channel_wise = True
            self._num_channels = len(threshold)

            if len(threshold) == 0:
                # can be dynamic only at init (before first data)
                # NB: _num_channles set dynamically - at first pass
                self._dynamic = True

        self.threshold = Parameter(torch.tensor(threshold))
        self.threshold.requires_grad = False
        self.inplace = inplace

    @property
    def dynamic(self) -> bool:
        return self._dynamic

    @property
    def channel_wise(self) -> bool:
        return self._channel_wise

    @property
    def num_channels(self):
        if self._dynamic:
            raise Exception(
                "number of channels not yet allocated. function should be called only after allocation"
            )

        return self._num_channels

    def set_threshold(self, threshold: Union[float, List[float]]):
        if self._dynamic:
            raise RuntimeError(
                "cannot set threshold, threshold is setup activation dynamic (constructor given empty list)"
            )

        if self._channel_wise and isinstance(threshold, float):
            raise ValueError(
                "cannot set threshold to float value, constructor setup with list of channels len {}".format(
                    self._num_channels
                )
            )

        if self._channel_wise and self._num_channels != len(threshold):
            raise ValueError(
                "cannot set threshold to list of len({}), constructor setup with list of len({})".format(
                    len(threshold), self._num_channels
                )
            )

        current_tens = self.threshold.data  # type: Tensor
        new_tens = current_tens.new_tensor(threshold)
        current_tens.copy_(new_tens)

    def get_threshold(self) -> Union[float, List[float]]:
        return (
            self.threshold.data.cpu().item()
            if not self._channel_wise
            else self.threshold.data.cpu().tolist()
        )

    def forward(self, inp: Tensor):
        if not self._channel_wise:
            threshold = self.threshold.data.item()

            return fat_relu(inp, threshold, self.inplace)

        if self._dynamic:
            thresh = [0.0] * inp.shape[1]
            self.threshold.data = torch.tensor(thresh)
            self._dynamic = False
            self._num_channels = len(thresh)

        assert (
            inp.shape[1] == self._num_channels
        )  # runtime test that #channels equals expected #channels

        return _apply_permuted_channels(
            fat_relu, inp, threshold=self.threshold, inplace=self.inplace
        )

    def extra_repr(self):
        inplace_str = ", inplace" if self.inplace else ""

        return "threshold={}{}".format(self.threshold, inplace_str)

    def load_state_dict(self, state_dict, strict=True):
        if self._dynamic:
            raise Exception(
                "attempt to load state_dict, but fatrelu is not initialized yet."
                "need to pass data once to initialize channel since constructed "
                "with dynamic allocation of number of channels"
            )

        super().load_state_dict(state_dict, strict)


class _DiffFATReLU(FATReLU):
    def __init__(
        self,
        threshold: Union[float, List[float]],
        clamp_thresh: Union[float, None],
        compression: Union[float, List[float]],
        clamp_comp: Union[float, None],
        inplace: bool = False,
    ):
        super().__init__(threshold, inplace)
        self._clamp_thresh = clamp_thresh
        self._clamp_comp = clamp_comp

        if type(threshold) != type(compression):
            raise Exception(
                "Type of threshold {} must match type of compression {}".format(
                    type(threshold), type(compression)
                )
            )

        if isinstance(compression, List) and len(compression) != len(threshold):
            raise Exception(
                "len of compression {} must match len of threshold {}".format(
                    len(compression), len(threshold)
                )
            )

        self.compression = Parameter(torch.tensor(compression))

    def set_compression(self, compression: Union[float, List[float]]):
        if self._dynamic:
            raise Exception(
                "attempted to set compression for a "
                "dynamic channel allocation - expected flow: pass input through model first"
            )

        if self._channel_wise and not isinstance(compression, List):
            raise ValueError(
                "attempted to set single compression, "
                "but expected channelized compression of length {}".format(
                    self._num_channels
                )
            )

        if self._channel_wise and self._num_channels != len(compression):
            raise ValueError(
                "attempted to set compression of len {} "
                "but number of channels was set to to {} "
                "at FATReLU init".format(len(compression), self._num_channels)
            )

        self._compression.data.copy_(torch.tensor(compression))

    def get_compression(self) -> Union[float, List[float]]:
        return (
            self._compression.data.cpu().item()
            if not self._channel_wise
            else self._compression.data.cpu().tolist()
        )

    # def _param_clamp(self):
    #     self.threshold.clamp(self.threshold_dynamic_val, 1e5)  # NB: not done inplace
    #     self.compression.clamp(self.compression_dynamic_val, 1e5)

    def forward(self, inp: Tensor):
        if self._clamp_thresh is not None:
            self.threshold.data.clamp_(self._clamp_thresh)

        if self._clamp_comp is not None:
            self.compression.data.clamp_(self._clamp_comp)

        if not self._channel_wise:
            return self.apply_diff_fat_relu(inp)

        if self._dynamic:
            # population of dynamic channel threshold and compression - only once: in first forward.
            thresh = [
                self._clamp_thresh if self._clamp_thresh is not None else 0.0
            ] * inp.shape[1]
            self.threshold.data = torch.tensor(thresh)

            comp = [
                self._clamp_comp if self._clamp_comp is not None else 1.0
            ] * inp.shape[1]
            self.compression.data = torch.tensor(comp)

            self._dynamic = False
            self._num_channels = len(thresh)

        return self.apply_diff_fat_relu(inp)

    @abstractmethod
    def apply_diff_fat_relu(self, inp: Tensor) -> Tensor:
        raise NotImplementedError()

    def extra_repr(self):
        inplace_str = "inplace" if self.inplace else ""

        return "threshold={}, compression={}{}".format(
            self.threshold, self.compression, inplace_str
        )


class FATPWReLU(_DiffFATReLU):
    def __init__(
        self,
        threshold: Union[float, List[float]],
        compression: Union[float, List[float]],
        clamp_thresh: bool = False,
        clamp_comp: bool = False,
        inplace: bool = False,
    ):
        """
        Applies a piece-wise linear approximation of the FAT ReLU such that the loss is now derivable
        according to the threshold and the slope.
        threshold=0 and compression=1 is a regular ReLU implementation

        f(x, t, c) = 0 if x <= (t - t/c)
                   = x if x >= t
                   = c(x - (t - t/c)) if x > (t - t/c) and x < t

        :param threshold: the threshold at which all values will be zero or interpolated between threshold and 0
        :param compression: the compression or slope to interpolate between 0 and the threshold with
        :param clamp_thresh: make sure the threshold cannot go below this value
        :param clamp_comp: make sure the compression cannot go below this value
        :param inplace: false to create a new tensor, true to overwrite the current tensor's values
        """
        clamp_thresh = 0.0 if clamp_thresh else None
        clamp_comp = 1.0 if clamp_comp else None
        super(FATPWReLU, self).__init__(
            threshold, clamp_thresh, compression, clamp_comp, inplace
        )

    def apply_diff_fat_relu(self, inp: Tensor) -> Tensor:
        out = _apply_permuted_channels(
            fat_pw_relu, inp, threshold=self.threshold, compression=self.compression
        )

        return out


class FATSigReLU(_DiffFATReLU):
    def __init__(
        self,
        threshold: Union[float, List[float]],
        compression: Union[float, List[float]],
        clamp_thresh: bool = False,
        clamp_comp: bool = False,
    ):
        """
        Applies a sigmoid approximation of the FAT ReLU such that the loss is now derivable
        according to the threshold and the slope

        f(x, t, c) = x / e^(c*(t-x))

        :param threshold: the threshold at which all values will be zero or approximated in the sigmoid region
        :param compression: the compression or slope to use in the sigmoid region
        :param clamp_thresh: make sure the threshold cannot go below this value
        :param clamp_comp: make sure the compression cannot go below this value
        """
        clamp_thresh = 0.0 if clamp_thresh else None
        clamp_comp = 0.0 if clamp_comp else None
        super(FATSigReLU, self).__init__(
            threshold, clamp_thresh, compression, clamp_comp, inplace=False
        )

    def apply_diff_fat_relu(self, inp: Tensor) -> Tensor:
        out = _apply_permuted_channels(
            fat_sig_relu, inp, threshold=self.threshold, compression=self.compression
        )

        return out


class FATExpReLU(_DiffFATReLU):
    def __init__(
        self,
        threshold: Union[float, List[float]],
        compression: Union[float, List[float]],
        clamp_thresh: bool = False,
        clamp_comp: bool = False,
    ):
        """
        Applies an exponential approximation of the FAT ReLU such that the loss is now derivable
        according to the threshold and the slope

        f(x, t, c) = 0 if x <= 0
                   = x if x >= t
                   = x * e^(c(x-t)) if x > 0 and x < t

        :param threshold: the threshold at which all values will be zero or approximated in the exponential region
        :param compression: the compression or slope to use in the exponential region
        :param clamp_thresh: make sure the threshold cannot go below this value
        :param clamp_comp: make sure the compression cannot go below this value
        """
        clamp_thresh = 0.0 if clamp_thresh else None
        clamp_comp = 1.0 if clamp_comp else None
        super(FATExpReLU, self).__init__(
            threshold, clamp_thresh, compression, clamp_comp, inplace=False
        )

    def apply_diff_fat_relu(self, inp: Tensor) -> Tensor:
        print(
            _apply_permuted_channels(
                fat_exp_relu,
                inp,
                threshold=self.threshold,
                compression=self.compression,
            )
        )
        out = _apply_permuted_channels(
            fat_exp_relu, inp, threshold=self.threshold, compression=self.compression
        )

        return out


class FATReluType(Enum):
    basic = "basic"
    piecewise = "piecewise"
    sigmoid = "sigmod"
    exponential = "exponential"


def convert_relus_to_fat(
    module: Module, type_: FATReluType = FATReluType.basic, **kwargs
) -> Dict[str, FATReLU]:
    relu_keys = []

    for name, mod in module.named_modules():
        if isinstance(mod, ReLU):
            relu_keys.append(name)

    added = {}

    for key in relu_keys:
        added[key] = set_relu_to_fat(module, key, type_, **kwargs)

    return added


def set_relu_to_fat(
    module: Module, layer_name: str, type_: FATReluType = FATReluType.basic, **kwargs
) -> FATReLU:
    if type_ == FATReluType.basic:
        construct = FATReLU
    elif type_ == FATReluType.piecewise:
        construct = FATPWReLU
    elif type_ == FATReluType.sigmoid:
        construct = FATSigReLU
    elif type_ == FATReluType.exponential:
        construct = FATExpReLU
    else:
        raise ValueError("unknown type_ given of {}".format(type_))

    layer = module
    layers = layer_name.split(".")

    for lay in layers[:-1]:
        layer = layer.__getattr__(lay)

    fat = layer.__getattr__(layers[-1])

    if not isinstance(fat, FATReLU):
        fat = construct(**kwargs)

    layer.__setattr__(layers[-1], fat)

    return fat
