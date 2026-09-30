from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")

texte = "Un ours en peluche mignon lit."

tokens_ids = tokenizer.encode(texte)

print(f"Texte initial: {texte}, Tokens découpés:")
for id in tokens_ids:
    print(f"{id}: {tokenizer.decode([id])}")

