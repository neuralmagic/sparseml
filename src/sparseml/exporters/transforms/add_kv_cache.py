import onnx.helper

from sparseml.exporters.transforms import OnnxTransform
from sparseml.exporters.transforms.utils import get_structural_matches
from onnx import ModelProto
from sparseml.onnx.utils import ONNXGraph, get_node_input_nodes, model_inputs


__all__ = ["AddKeyValueCache"]


class AddKeyValueCache(OnnxTransform):
    def transform(self, model: ModelProto) -> ModelProto:
        self.pre_validate(model)
        graph = ONNXGraph(model)

        # overwrite mask input
        inputs = model_inputs(model)
        attention_mask_input = inputs[1]

        # overwrite the lenght of mask input
        attention_mask_input.type.tensor_type.shape.dim[1].dim_param = "sequence_len + 1"

        value_matches = get_structural_matches(
            graph,
            op_type="MatMul",
            parent_ops=[
                    ["Softmax"],
                ],

        )

        key_matches = get_structural_matches(
            graph,
            op_type="MatMul",
            children_ops=[
                    ["Reshape", "Add", "Max", "Reshape", "Softmax"],
                ],

        )

        for i, match in enumerate(key_matches):
            self.log_match(match)
            match_input_nodes = get_node_input_nodes(model, match.node)

            key_node = [node for node in match_input_nodes if node.op_type != "Reshape"][0]
            index = [i for i, n in enumerate(match.node.input) if "MatMul" in n][1] # hack, careful!
            match.node.input[index] = f"placeholder_key_{i}"

            # create input node
            new_input = onnx.helper.make_tensor_value_info(
                f"past_key_{i}", onnx.TensorProto.FLOAT, ["batch * num_heads", "hidden_dims", "sequence_len"]
            )

            new_node = onnx.helper.make_node(op_type="Concat",
                                             inputs=[key_node.output[0], f"past_key_{i}", ],
                                             outputs=[f"placeholder_key_{i}"],
                                             axis=2,
                                             name=f"concat_key_{i}")

            model.graph.node.insert([i for i,n in enumerate(graph._model.graph.node) if n.name ==match.node.name][0], new_node)
            model.graph.input.insert(-1, new_input)
            self.add_node_deferred(new_node)

        for i, match in enumerate(value_matches):
            self.log_match(match)
            match_input_nodes = get_node_input_nodes(model, match.node)

            # get the value node and rename the output
            value_node = [node for node in match_input_nodes if node.op_type != "Softmax"][0]
            index = [i for i,n in enumerate(match.node.input) if "MatMul" in n][0]
            match.node.input[index] = f"placeholder_value_{i}"


            # create input node
            new_input = onnx.helper.make_tensor_value_info(
                f"past_value_{i}", onnx.TensorProto.FLOAT, ["batch * num_heads", "sequence_len", "hidden_dims"]
            )

            new_node = onnx.helper.make_node(op_type="Concat",
                                             inputs=[f"past_value_{i}", value_node.output[0]],
                                             outputs=[f"placeholder_value_{i}"],
                                             axis=1,
                                             name=f"concat_value_{i}")

            model.graph.node.insert([i for i,n in enumerate(graph._model.graph.node) if n.name ==match.node.name][0], new_node)
            model.graph.input.insert(-1, new_input)
            self.add_node_deferred(new_node)

        return model # onnx::MatMul_617