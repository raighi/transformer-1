from transformers import pipeline

unmasker = pipeline("fill-mask",model="bert-base-uncased")
phrase = "I am a fucking [MASK] and you know it Mary Jane."
resultats = unmasker(phrase)

for res in resultats:
    print(f"Score:{res['score']:.3f} | Mot: {res['token_str']} | Phrase: {res['sequence']}")