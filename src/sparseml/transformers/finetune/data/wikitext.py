from sparseml.transformers.finetune.data import TextGenerationDataset

@TextGenerationDataset.register(name="wikitext")
class WikiTextDataset(TextGenerationDataset):
    def __init__(self, data_args, tokenizer):
        super().__init__(
            text_column="text",
            data_args=data_args,
            tokenizer=tokenizer
        )

