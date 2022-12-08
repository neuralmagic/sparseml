from sparseml.exporters.transforms import OnnxTransform
from onnx import ModelProto
from sparseml.onnx.utils import ONNXGraph
from sparseml.exporters.utils.matching import get_structural_matches


class DeleteTrivialOnnxAdds(OnnxTransform):

    def transform(self, model: ModelProto) -> ModelProto:
        count_converted_nodes = 0
        graph = ONNXGraph(model)
        for match in get_structural_matches(
                graph,
                op_type="Add",
        ):
            _LOGGER.debug(f"Matched Identity node: {match.node.name}")
            model = fold_identity_initializer(match, model)
            count_converted_nodes += 1

        if count_converted_nodes > 0:
            _LOGGER.info(f"Folded {count_converted_nodes} identity initializer nodes")
        return model

def _delete_trivial_onnx_adds(model: onnx.ModelProto):
    # delete all add nodes in the graph with second inputs as constant nodes set to 0
    add_nodes = [node for node in model.graph.node if node.op_type == "Add"]
    for add_node in add_nodes:
        try:
            add_const_node = [
                node for node in model.graph.node if node.output[0] == add_node.input[1]
            ][0]
            add_const_val = numpy_helper.to_array(add_const_node.attribute[0].t)
            if numpy.all(add_const_val == 0.0):
                # update graph edges
                parent_node = [
                    node
                    for node in model.graph.node
                    if add_node.input[0] in node.output
                ]
                if not parent_node:
                    continue
                parent_node[0].output[0] = add_node.output[0]
                # remove node and constant
                model.graph.node.remove(add_node)
                model.graph.node.remove(add_const_node)
        except Exception:  # skip node on any error
            continue