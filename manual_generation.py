import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")

model_causal = AutoModelForCausalLM.from_pretrained("gpt2")

texte_initial = "Once upon a time"

input_ids = tokenizer.encode(texte_initial, return_tensors="pt")

max_new_tokens = 10
print("Génération de texte étape par étape :")
for step in range(max_new_tokens):
    with torch.no_grad():
        outputs = model_causal(input_ids)
        next_token_logits = outputs.logits[0, -1,:]
        next_token_id = torch.argmax(next_token_logits).unsqueeze(0).unsqueeze(0)
        input_ids = torch.cat([input_ids, next_token_id], dim=-1)
        print(f"Etape {step+1}: {tokenizer.decode(input_ids[0])}")