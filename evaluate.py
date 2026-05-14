import re

def normalize(text):
    text = text.lower()
    text = re.sub(r"\W+", " ", text)
    return text.strip()

def exact_match(pred, gold_list):
    pred = normalize(pred)
    for g in gold_list:
        if normalize(g) == pred:
            return 1
    return 0

def compute_em(predictions, golds):
    return sum(exact_match(p, g) for p, g in zip(predictions, golds)) / len(predictions)