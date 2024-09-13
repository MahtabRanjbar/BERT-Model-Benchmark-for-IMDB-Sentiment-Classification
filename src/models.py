from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODELS = [
    "bert-base-cased",
    "albert-base-v2",
    "roberta-base", 
    "distilbert-base-uncased"
]


def load_model_and_tokenizer(model_name, num_labels=2):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    return model, tokenizer