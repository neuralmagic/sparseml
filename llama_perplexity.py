from deepsparse.transformers.evaluator import TransformersEvaluator


model = "/home/sadkins/sparseml/deploy_llama/deployment"
dataset = "wikitext2"
evaluator = TransformersEvaluator(model=model, dataset=dataset)
evaluator.evaluate()
print(evaluator.get_results())