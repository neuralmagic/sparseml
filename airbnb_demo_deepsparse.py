### DeepSparse Inference
from transformers import AutoConfig
from datasets import load_dataset
from deepsparse import Pipeline

dataset_name = "tweet_eval"
dataset_subname = "sentiment"
dataset_test = load_dataset(dataset_name, dataset_subname, split="test").shuffle(seed=420)
config = AutoConfig.from_pretrained("./oneshot_deployment/deployment")
pipeline = Pipeline.create("text-classification", model_path = "./oneshot_deployment/deployment", engine_type="onnxruntime")
correct_labels = 0
for i, sample in enumerate(dataset_test):
    text, label = sample["text"], sample["label"]
    out = pipeline(text)
    print(out)
    pred_label = config.label2id[out.labels[0]]
    correct_labels += label == pred_label
    print(f"% of correct predictions: {(correct_labels/(i+1))*100}")
    if i == 50:
        break
    