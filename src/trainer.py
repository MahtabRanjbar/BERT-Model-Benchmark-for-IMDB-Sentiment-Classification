import numpy as np
from datasets import load_metric
from sklearn.metrics import precision_recall_fscore_support
from transformers import Trainer, TrainingArguments
from transformers import EarlyStoppingCallback


def compute_metrics(eval_pred):
    accuracy_metric = load_metric("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')

    return {
        "accuracy": accuracy["accuracy"],
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def create_trainer(model, train_dataset, eval_dataset, compute_metrics, config):
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=config['training']['batch_size'],
        per_device_eval_batch_size=config['evaluation']['batch_size'],
        num_train_epochs=config['training']['num_epochs'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=config['checkpointing']['monitor'],
        greater_is_better=(config['checkpointing']['mode'] == 'max'),
        push_to_hub=False,
        warmup_steps=config['scheduler']['num_warmup_steps'],
        lr_scheduler_type="linear",
    )

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=config['early_stopping']['patience'],
        early_stopping_threshold=config['early_stopping']['min_delta']
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping_callback],
    )

    return trainer

