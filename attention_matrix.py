import torch
import matplotlib.pyplot as plt 
import seaborn as sns
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModel.from_pretrained("gpt2", output_attentions=True)

sentence = "The animal didn't cross the street because it was too tired."
inputs = tokenizer(sentence, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

attention_matrix = outputs.attentions[-1][0,0].numpy()

tokens_str = [tokenizer.decode([tok]) for tok in inputs["input_ids"][0]]

plt.figure(figsize=(10,8))
sns.heatmap(attention_matrix, xticklabels=tokens_str, yticklabels=tokens_str, cmap="viridis")

plt.title("Matrice d'attention (Dernière couche, tête 0)")
plt.xlabel("Key Tokens (mots regardés)")
plt.ylabel("Query tokens (mots qui regardent)")
plt.show()