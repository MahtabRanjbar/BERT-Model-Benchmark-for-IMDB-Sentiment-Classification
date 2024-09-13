import pandas as pd
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from data_processing import tokenize_dataset
from models import load_model_and_tokenizer


def run_benchmark(models, train_dataset, eval_dataset, create_trainer, compute_metrics,config):
    results = []

    for model_name in tqdm(models, desc="Benchmarking models"):
        model, tokenizer = load_model_and_tokenizer(model_name)

        tokenized_train = tokenize_dataset(train_dataset, tokenizer)
        tokenized_eval = tokenize_dataset(eval_dataset, tokenizer)

        trainer = create_trainer(model, tokenized_train, tokenized_eval, compute_metrics,config)

        train_result = trainer.train()
        eval_result = trainer.evaluate()

        num_parameters = sum(p.numel() for p in model.parameters())

        results.append({
            "model": model_name,
            "accuracy": eval_result["eval_accuracy"],
            "precision": eval_result["eval_precision"],
            "recall": eval_result["eval_recall"],
            "f1": eval_result["eval_f1"],
            "train_runtime": train_result.metrics["train_runtime"],
            "num_parameters": num_parameters
        })

    return pd.DataFrame(results)


def plot_benchmark_results(df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    df.plot(x="model", y="accuracy", kind="bar", ax=ax1)
    ax1.set_title("Model Accuracy")
    ax1.set_ylabel("Accuracy")

    df.plot(x="model", y="num_parameters", kind="bar", ax=ax2)
    ax2.set_title("Model Size")
    ax2.set_ylabel("Number of Parameters")

    plt.tight_layout()
    plt.savefig("../results/benchmark_results.png")
    plt.close()


def save_benchmark_results(df):
    df.to_csv("../results/benchmark_results.csv", index=False)