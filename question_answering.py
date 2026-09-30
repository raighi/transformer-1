from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer
import torch


nom_modele = "deepset/bert-base-cased-squad2"
tokenizer = AutoTokenizer.from_pretrained(nom_modele)
model = AutoModelForQuestionAnswering.from_pretrained(nom_modele)

contexte = """
IMT Atlantique is a French Graduate school located in Brittany.
The embedded systems curriculum allows student to acquire the necessary skills to design and develop embedded systems.
It is taught in the Department of Computer Science and Networking, which is part of the School of Engineering."""

question = "In which department is the embedded systems curriculum taught?"


inputs = tokenizer(question, contexte, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

start_index = torch.argmax(outputs.start_logits)
end_index = torch.argmax(outputs.end_logits)+1
reponse_tokens = inputs["input_ids"][0][start_index:end_index]
reponse_texte = tokenizer.decode(reponse_tokens)

print("Réponse extraite :",reponse_texte)