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

"""
Constants related to sparseml pytorch quantization flows
"""


__all__ = [
    "FUSED_MODULE_NAMES",
    "NON_QUANTIZABLE_MODULE_NAMES",
]


"""
Quantization Modifier quantizes all 'leaf' level modules by default
this list contains modules that are very unlikely to be desired for quantization
and will not have a QuantizationScheme ever attached to them by the modifier
QuantizationSchemes may be manually attached with the .quantization_scheme
property and the modifier will then pick up the module for quantization
"""
NON_QUANTIZABLE_MODULE_NAMES = {
    # no-ops
    "Module",
    "Identity",
    "Flatten",
    "Unflatten",
    "DataParallel",
    "ModuleList",
    "Sequential",
    # losses
    "L1Loss",
    "NLLLoss",
    "KLDivLoss",
    "MSELoss",
    "BCELoss",
    "BCEWithLogitsLoss",
    "NLLLoss2d",
    "PoissonNLLLoss",
    "CosineEmbeddingLoss",
    "CTCLoss",
    "HingeEmbeddingLoss",
    "MarginRankingLoss",
    "MultiLabelMarginLoss",
    "MultiLabelSoftMarginLoss",
    "MultiMarginLoss",
    "SmoothL1Loss",
    "GaussianNLLLoss",
    "HuberLoss",
    "SoftMarginLoss",
    "CrossEntropyLoss",
    "TripletMarginLoss",
    "AdaptiveLogSoftmaxWithLoss",
    "TripletMarginWithDistanceLoss",
    # dropouts
    "Dropout",
    "Dropout1d",
    "Dropout2d",
    "Dropout3d",
    "AlphaDropout",
    "FeatureAlphaDropout",
}


FUSED_MODULE_NAMES = {
    # Conv based layers
    "ConvBn1d",
    "ConvBn2d",
    "ConvBn3d",
    "ConvReLU1d",
    "ConvReLU2d",
    "ConvReLU3d",
    "ConvBnReLU1d",
    "ConvBnReLU2d",
    "ConvBnReLU3d",
    # Linear Layers
    "LinearReLU",
}
