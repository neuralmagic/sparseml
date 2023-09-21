from sparseml.transformers.export import MODEL_ONNX_NAME, export

model_path = "/home/sadkins/sparseml/deploy_llama/obcq_deployment"
sequence_length = 384
onnx_file_name = MODEL_ONNX_NAME
trust_remote_code = True

export(
    task="text-generation",
    model_path=model_path,
    sequence_length=sequence_length,
    no_convert_qat=True,
    finetuning_task=None,
    onnx_file_name=onnx_file_name,
    num_export_samples=0,
    trust_remote_code=trust_remote_code,
    data_args=None,
    one_shot=None,
)

from deepsparse.transformers.evaluator import TransformersEvaluator