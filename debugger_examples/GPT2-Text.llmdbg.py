"""Start here: inspect attention, MLPs, residuals, and next-token predictions."""
from llmdbg import debug, gpt2, nop

model = gpt2(
    "openai-community/gpt2",
    revision="main",
    device="auto",
    dtype="auto",
)

prompt = "Paris is the capital and largest city of France. It stands on the Seine and has been a major center of finance, diplomacy, commerce, culture, fashion, and gastronomy since the seventeenth century. Visitors often recognize the city by the Eiffel Tower and"
tokens = model.tokenize(prompt)


def attention(x, block):
    normalized = block.layernorm(x)
    qkv = block.qkv_projection(normalized)
    mixed = block.attention_mixing(qkv)
    update = block.output_projection(mixed)
    return x + update


def mlp(x, block):
    normalized = block.layernorm(x)
    hidden = block.expansion(normalized)
    hidden = block.gelu(hidden)
    update = block.projection(hidden)
    return x + update


with debug(model, tokens) as run:
    for token in run.tokens:
        x = model.token_embedding(tokens)
        for layer, block in enumerate(model.layers):
            nop()  # resid-pre
            x = attention(x, block)
            nop()  # resid-mid
            x = mlp(x, block)
            nop()  # resid-post

        x = model.final_layernorm(x)
        logits = model.vocabulary_projection(x)
        predictions = model.top_tokens(logits, k=10)
        nop()
