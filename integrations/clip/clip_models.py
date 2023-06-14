import torch
import torch.nn as nn


class VisualModel(nn.Module):
    def __init__(self, visual_model, transformations):

        super().__init__()

        self.visual_model = visual_model
        self.transformations = transformations

    def forward(self, x):
        return self.visual_model(x)


class TextModel(nn.Module):
    def __init__(
        self,
        token_embedding,
        tokenizer,
        positional_embedding,
        transformer,
        ln_final,
        text_projection,
        attn_mask,
    ):

        super().__init__()

        self.token_embedding = token_embedding
        self.tokenizer = tokenizer
        self.positional_embedding = positional_embedding
        self.transformer = transformer
        self.ln_final = ln_final
        self.text_projection = text_projection
        self.attn_mask = attn_mask
        self.cast_dtype = self.transformer.get_cast_dtype()

    def forward(self, input_ids):
        x = self.token_embedding(input_ids).to(self.cast_dtype)
        x = x + self.positional_embedding.to(self.cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), input_ids.argmax(dim=-1)] @ self.text_projection
        return x
