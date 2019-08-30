import os
import importlib
import torch


__all__ = ['fix_onnx_threshold_export']


def fix_onnx_threshold_export():
    # Required to fix pytorch onnx export so it will export a thresholded relu (FATReLU) properly
    module_path = os.path.realpath(torch.onnx.__file__)
    module_dir = os.path.dirname(module_path)

    if os.path.exists(os.path.join(module_dir, 'symbolic.py')):
        # older code
        sym_path = os.path.join(module_dir, 'symbolic.py')

        with open(sym_path, 'r') as symbolic_file:
            code = symbolic_file.read()

        code = code.replace('return _unimplemented("threshold", "non-zero threshold")',
                            "return g.op('ThresholdedRelu', self, alpha_f=_scalar(threshold))")

        with open(sym_path, 'w') as symbolic_file:
            symbolic_file.write(code)
    elif os.path.exists(os.path.join(module_dir, 'symbolic_opset9.py')):
        sym_path = os.path.join(module_dir, 'symbolic_opset9.py')

        with open(sym_path, 'r') as symbolic_file:
            code = symbolic_file.read()

        code = code.replace('return _unimplemented("threshold", "non-zero threshold")',
                            "return g.op('ThresholdedRelu', self, alpha_f=sym_help._scalar(threshold))")

        with open(sym_path, 'w') as symbolic_file:
            symbolic_file.write(code)
    else:
        raise Exception('could not find the onnx symbolicy file to fix threshold for')

    importlib.reload(torch.onnx)
