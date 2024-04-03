import polars as pl
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from transformers import AutoModel, AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
import datasets
import torch
import numpy as np
from datasets import DatasetDict, Dataset


class Modeling:
    def __init__(self):
        self.num_labels = 2
        self.batch_size = 16
        self.model_name = "luba-kaggle-tfug-toxic-clf"
        self.df_train = pl.read_csv("data/train.csv")
        self.df_test = pl.read_csv("data/test.csv")
        self.df_train_clean = self._clean_data(self.df_train)
        self.dataset_all = self._create_dataset_all()
        self.logging_steps = len(self.df_train_clean) // self.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(
            "neuralmind/bert-large-portuguese-cased"
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "neuralmind/bert-large-portuguese-cased", num_labels=self.num_labels
        ).to("cuda:1")

    def _create_dataset_all(self):
        train, test = train_test_split(
            self.df_train_clean.to_pandas(), test_size=0.2, random_state=42
        )

        train_dataset = Dataset.from_pandas(train)
        val_dataset = Dataset.from_pandas(test)
        test_dataset = Dataset.from_pandas(self.df_test.to_pandas())
        return DatasetDict(
            {"train": train_dataset, "val": val_dataset, "test": test_dataset}
        )

    def _compute_metrics(self, pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        f1 = f1_score(labels, preds, average="weighted")
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "f1": f1}

    def _clean_data(self, df: pl.DataFrame):
        return df.drop_nulls("label").unique("text")

    def _tokenize(self, batch):
        return self.tokenizer(
            batch["text"],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

    def _extract_hidden_states(self, batch):
        inputs = {
            k: v.to("cuda:1")
            for k, v in batch.items()
            if k in self.tokenizer.model_input_names
        }

        with torch.no_grad():
            last_hidden_states = self.model(**inputs).last_hidden_state

        return {"hidden_states": last_hidden_states[:, 0].cpu().numpy()}

    def encoding(self):
        dataset_encoded = self.dataset_all.map(
            self._tokenize, batched=True, batch_size=None
        )
        return dataset_encoded

    def train(self, dataset_encoded):
        training_args = TrainingArguments(
            output_dir=self.model_name,
            num_train_epochs=10,
            learning_rate=2e-5,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            disable_tqdm=False,
            logging_steps=self.logging_steps,
            push_to_hub=False,
            log_level="error",
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            compute_metrics=self._compute_metrics,
            train_dataset=dataset_encoded["train"],
            eval_dataset=dataset_encoded["val"],
            tokenizer=self.tokenizer,
        )

        trainer.train()
        self.trainer = trainer

    def predict(self, dataset_encoded):
        preds = self.trainer.predict(dataset_encoded["test"])
        y_preds = np.argmax(preds.predictions, axis=1)

        submission = pl.DataFrame({"id": self.df_test["id"], "label": y_preds})
        submission.write_csv("data/submission_02.csv")


if __name__ == "__main__":
    modeling = Modeling()
    dataset_encoded = modeling.encoding()

    # train model
    modeling.train(dataset_encoded)

    # test
    modeling.predict(dataset_encoded)
