from datasets import load_dataset


def load_imdb_data():
    return load_dataset("imdb")


def prepare_dataset(dataset, num_samples=1000):
    train_dataset = dataset["train"].shuffle(seed=42).select(range(num_samples))
    eval_dataset = dataset["test"].shuffle(seed=42).select(range(num_samples))
    
    return train_dataset, eval_dataset


def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512,  # You can adjust this value based on your needs
        return_tensors="pt"
    )


def tokenize_dataset(dataset, tokenizer):
    tokenized = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
    )
    tokenized = tokenized.remove_columns(["text"])
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch")
    return tokenized