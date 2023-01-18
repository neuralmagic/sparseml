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
Required to avoid running into
circular dependencies when importing:

isort:skip_file
"""

# flake8: noqa
from .base_transform import *
from .onnx_transform import *

from .delete_trivial_onnx_adds import DeleteTrivialOnnxAdds
from .delete_repeated_qdq import DeleteRepeatedQdq
from .fold_identity_initializers import FoldIdentityInitializers
from .quantize_qat_embedding import QuantizeQATEmbedding
from .matmul_to_qlinearmatmul import MatMulToQLinearMatMul
from .conv_to_convinteger_add_cast_mul import ConvToConvIntegerAddCastMul
from .flatten_qparams import FlattenQParams
from .unwrap_batchnorms import UnwrapBatchNorms
from .constants_to_initializers import ConstantsToInitializers
from .conv_to_qlinearconv import ConvToQLinearConv
from .conv_to_convinteger_add_cast_mul import ConvToConvIntegerAddCastMul
from .fold_conv_div_bn import FoldConvDivBn
from .fold_relu_quants import FoldReLUQuants
from .gemm_to_matmulinteger_add_cast_mul import GemmToMatMulIntegerAddCastMul
from .gemm_to_qlinearmatmul import GemmToQLinearMatMul
from .initializers_to_uint8 import InitializersToUint8
from .matmul_to_matmulinteger_add_cast_mul import MatMulToMatMulIntegerAddCastMul
from .propagate_embedding_quantization import PropagateEmbeddingQuantization
from .quantize_qat_embedding import QuantizeQATEmbedding
from .quantize_residuals import QuantizeResiduals
from .remove_duplicate_qconv_weights import RemoveDuplicateQConvWeights
from .remove_duplicate_quantize_ops import RemoveDuplicateQuantizeOps
from .skip_input_quantize import SkipInputQuantize
