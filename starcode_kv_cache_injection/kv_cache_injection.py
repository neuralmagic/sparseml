from transformers import AutoTokenizer, AutoConfig

import onnx
import logging
import os
from typing import List, Optional
from onnx import TensorProto, ModelProto, helper, NodeProto
from sparseml.onnx.utils import ONNXGraph
from sparseml.exporters.transforms.kv_cache.cache_keys_and_values import reshape_kv_cache_inputs_outputs
from sparseml.exporters.transforms.kv_cache.transforms_codegen import AdditionalTransformsCodeGen
from sparseml.onnx.utils.helpers import get_nodes_by_output_id

_LOGGER = logging.getLogger(__name__)

class AdditionalTransformsBigCode(AdditionalTransformsCodeGen):
    """
    Since the entries of the causal mask are similar in their values
    and layout to the CodeGen causal mask, I inherit from the
    AdditionalTransformsCodeGen class
    """

    # position ids are created by a Sub node (the one that is folllowed by a Where node
    # in the onnx graph)
    POSITION_IDS_MATCHING_PATTERN = dict(op_type="Sub", children_ops=[["Where"]])
    # causal mask is created by a Unsqueeze node (the one that is folllowed by a Where node
    # in the onnx graph)
    CAUSAL_MASK_MATCHING_PATTERN = dict(op_type="Unsqueeze", children_ops=[["Where", "Softmax"]])

    def swap_nodes_for_input(
        self,
        model: ModelProto,
        nodes: List[NodeProto],
        input_name: str,
        nodes_parent_op_type: Optional[str] = None,
    ) -> ModelProto:

        """
        Injects the specified input to the graph, replacing the specified nodes.

        :param model: the ONNX model to inject the input into
        :param nodes: the nodes to replace with the input
        :param input_name: the name of the input to replace the nodes with
        :param nodes_parent_op_type: the parent op type of the nodes to replace

        :return: the updated model
        """

        graph = ONNXGraph(model)
        for node in nodes:
            # edits so that we can have multiple children nodes
            children_nodes = graph.get_node_children(node)
            for child_node in children_nodes:
                if nodes_parent_op_type:
                    assert child_node.op_type == nodes_parent_op_type, (
                        f"Expected to find {nodes_parent_op_type} node, "
                        f"found {child_node.op_type}"
                    )
                output_to_replace = node.output[0]
                self.log_match(node)
                for idx, input_name_child_node in enumerate(child_node.input):
                    if input_name_child_node == output_to_replace:
                        graph.update_node_input(child_node, input_name, idx)

        graph.delete_orphaned_node_branches()

        _LOGGER.info(
            f"Successfully swapped {len(nodes)} nodes for input '{input_name}'"
        )

        return model

    def add_constant_reshape_node(self, model: ModelProto) -> ModelProto:
        """
        Adds positions as an input to the model.

        Positions is a tensor of shape and dtype
        equal to input_ids.

        :param model: model to update
        :return: updated model
        """
        graph = ONNXGraph(model)
        # create a constant node that will feed value (1, 256, 768) to the reshape node
        constant_node = onnx.helper.make_node(
            "Constant",
            inputs=[],
            name="abc",
            outputs=["reshape_input"],
            value=onnx.helper.make_tensor(
                name="const_tensor",
                data_type=TensorProto.INT64,
                dims=[3],
                vals=[1, 256, 768],
            ),
        )
        graph.add_node(constant_node)
        reshape_node = get_nodes_by_output_id(model, "/transformer/Reshape_2_output_0")[0]
        reshape_node.input[1] = "reshape_input"
        _LOGGER.info(f"Inserted constant reshape node to the ONNX model")
        return model
    
    def add_causal_mask_reshape_node(self, model: ModelProto) -> ModelProto:
        """
        Adds positions as an input to the model.

        Positions is a tensor of shape and dtype
        equal to input_ids.

        :param model: model to update
        :return: updated model
        """
        graph = ONNXGraph(model)

        transpose_node = onnx.helper.make_node(
        op_type="Transpose",
        inputs=["causal_mask"],
        outputs=["causal_mask_transpose"],
        name=f"causal_mask_transpose",
        perm=(0,3,2,1),
    )
        graph.add_node(transpose_node)
        reshape_node = get_nodes_by_output_id(model, "causal_mask_adjusted")[0]
        reshape_node.input[0] = "causal_mask_transpose"
        _LOGGER.info(f"Inserted transpose to the causal mask in  the ONNX model")
        return model

    def transform(self, model: ModelProto) -> ModelProto:
        """
        1. Adds `positions` as an input to the model
        2. Adds `causal_mask` as an input to the model
        2. Finds the node that initially creates the `position_ids` tensor
        3. Updates the node to use the positions input instead of
           computing it from the Range op
        4. Finds the nodes that initially create the `causal_mask` tensors
        5. Updates the nodes to use the causal_mask input instead of
              computing it from the Slice op

        :param model: model to update
        :return: updated model
        """
        model = self.add_positions_input(model)
        model = self.add_causal_mask_input(model)
        model = self.add_constant_reshape_node(model)


        position_ids_nodes = self.find_nodes_by_pattern(
            model, pattern=self.POSITION_IDS_MATCHING_PATTERN
        )
        if len(position_ids_nodes) != 1:
            raise ValueError(
                "Expected to find exactly one node matching "
                f"the pattern {self.POSITION_IDS_MATCHING_PATTERN}, "
                f"found {len(position_ids_nodes)}"
            )

        model = self.inject_positions(model, position_ids_nodes, "Where")

        causal_mask_nodes = self.find_nodes_by_pattern(
            model, pattern=self.CAUSAL_MASK_MATCHING_PATTERN
        )
        model = self.inject_causal_mask(model, causal_mask_nodes, "Where")
        model = self.adjust_causal_mask(model)
        model = self.add_causal_mask_reshape_node(model)
        return model

def inject_kv_cache_inputs_outputs(model: ModelProto, names_nodes: List[str], hidden_size_kv_cache, batch_size = 1, key: bool = True, output_num:int=0):
    graph = ONNXGraph(model)

    inputs_to_add = []
    outputs_to_add = []
    num_attention_heads = 1
    attention_layer_idx = 0

    for node in model.graph.node:
        if node.name in names_nodes:

            # inject kv cache input/output
            cache_name = "key" if key else "value"
            cache_input_name_concat = f"past_key_values.{attention_layer_idx}.{cache_name}"
            cache_output_name_concat = f"present.{attention_layer_idx}.{cache_name}"

            cache_input_info = onnx.helper.make_tensor_value_info(
                cache_input_name_concat,
                TensorProto.FLOAT,
                [
                    batch_size,
                    num_attention_heads,
                    "past_sequence_len",
                    hidden_size_kv_cache,
                ]
            )

            cache_output_info = onnx.helper.make_tensor_value_info(
                cache_output_name_concat,
                TensorProto.FLOAT,
                [
                    batch_size,
                    num_attention_heads, 
                    "past_sequence_len + 1",
                    hidden_size_kv_cache,
                ]
            )

            model, cache_input_dims_concat, cache_input_name_concat, cache_output_name_concat = reshape_kv_cache_inputs_outputs(
                model=model,
                cache_input_name=cache_input_name_concat,
                cache_output_name=cache_output_name_concat,
                cache_input_dims= [
                    batch_size,
                    num_attention_heads,
                    "past_sequence_len",
                    hidden_size_kv_cache,
                ],
                batch_size=batch_size,
                num_attention_heads=1,
            )
            cache_parent = node
            concat_axis = 1 # concat over length axis
            concat_node = onnx.helper.make_node(
                op_type="Concat",
                inputs=[cache_input_name_concat, cache_parent.output[output_num]],
                outputs=[cache_output_name_concat],
                axis=concat_axis,
                name=f"concat.{cache_input_name_concat}",
            )

            for _node in model.graph.node:
                for input_idx, input_id in enumerate(_node.input):
                    if input_id == cache_parent.output[output_num] and _node.name != concat_node.name:
                        _node.input[input_idx] = cache_output_name_concat

            graph.add_node(concat_node)
            inputs_to_add.extend([cache_input_info])
            outputs_to_add.extend([cache_output_info])

            attention_layer_idx += 1
            print(f"Injected kv cache input/output for {attention_layer_idx}:{cache_name}")

    model.graph.input.extend(inputs_to_add)
    model.graph.output.extend(outputs_to_add)
    return model


def main(deployment_folder_path, save_name_injected_model):
    onnx_model = onnx.load(os.path.join(deployment_folder_path, "model.onnx"), load_external_data=False)
    config = AutoConfig.from_pretrained(os.path.join(deployment_folder_path, "config.json"))
    # KV Cache injection
    onnx_model = inject_kv_cache_inputs_outputs(model = onnx_model,
                                                names_nodes=[f"/transformer/h.{i}/attn/Split_1" for i in range(config.n_layer)],
                                                hidden_size_kv_cache= config.n_embd // config.n_head,
                                                key=True,
                                                output_num=0)
    onnx_model = inject_kv_cache_inputs_outputs(model = onnx_model,
                                                names_nodes=[f"/transformer/h.{i}/attn/Split_1" for i in range(config.n_layer)],
                                                hidden_size_kv_cache= config.n_embd // config.n_head,
                                                key=False,
                                                output_num=1)
    # Adjustment of causal masks and positions
    transformation = AdditionalTransformsBigCode()
    onnx_model = transformation.transform(model = onnx_model)
    # Save the model
    _LOGGER.info(f"Saved injected model to {os.path.join(deployment_folder_path, save_name_injected_model)}")
    onnx.save_model(onnx_model, os.path.join(deployment_folder_path, save_name_injected_model))



if __name__ == "__main__":
    PATH_TO_DEPLOYMENT_FOLDER = "/Users/damian/Code/nm/sparseml/tiny_starcoder_py/deployment/"
    # model created by running:
    # sparseml.export /Users/damian/Code/nm/sparseml/tiny_starcoder_py/ --task text-generation --integration transformers  --sequence_length 256 --trust_remote_code True
    NAME_INJECTED_MODEL = "test.onnx"
    main(PATH_TO_DEPLOYMENT_FOLDER, NAME_INJECTED_MODEL)


