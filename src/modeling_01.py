import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from transformers import AutoModel, AutoTokenizer
import datasets
import torch
import numpy as np


class Modeling:
    def __init__(self):
        self.df_train = pl.read_csv("data/train.csv")
        self.df_test = pl.read_csv("data/test.csv")
        self.df_train_clean = self._clean_data(self.df_train)
        self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
        self.model = AutoModel.from_pretrained("xlm-roberta-large").to("cuda")

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
            k: v.to("cuda")
            for k, v in batch.items()
            if k in self.tokenizer.model_input_names
        }

        with torch.no_grad():
            last_hidden_states = self.model(**inputs).last_hidden_state

        return {"hidden_states": last_hidden_states[:, 0].cpu().numpy()}

    def encoding(self, df: pl.DataFrame):
        dataset = datasets.Dataset.from_pandas(df.to_pandas())
        dataset_tokenized = dataset.map(self._tokenize, batched=True, batch_size=None)
        try:
            dataset_tokenized.set_format(
                type="torch", columns=["input_ids", "attention_mask", "label"]
            )
        except ValueError:
            dataset_tokenized.set_format(
                type="torch", columns=["input_ids", "attention_mask"]
            )

        dataset_hidden = dataset_tokenized.map(
            self._extract_hidden_states, batched=True
        )

        X_train = np.array(dataset_hidden["hidden_states"])
        try:
            y_train = np.array(dataset_hidden["label"])

            return X_train, y_train
        except KeyError:
            return X_train

    def train(self, X_train, y_train, X_test, y_test):
        model_spec = {
            "lr": LogisticRegression(max_iter=3000, random_state=42, n_jobs=-1),
            "hist": HistGradientBoostingClassifier(random_state=42),
        }

        for key, value in model_spec.items():
            model = value.fit(X_train, y_train)
            print(f"{key} score: {model.score(X_test, y_test)}")

            # not necessary the best model will be saved
            self.classifier = model

    def predict(self, X_test, df_test: pl.DataFrame):
        predictions = self.classifier.predict(X_test)

        submission = pl.DataFrame({"id": df_test["id"], "label": predictions})
        submission.write_csv("data/submission.csv")


def run():
    modeling = Modeling()
    X_train_encoded = modeling.encoding(modeling.df_train_clean)

    # split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_train_encoded[0], X_train_encoded[1], test_size=0.2, random_state=42
    )

    # train model
    modeling.train(X_train, y_train, X_test, y_test)

    # test
    X_test_encoded = modeling.encoding(modeling.df_test)

    modeling.predict(X_test_encoded, modeling.df_test)


if __name__ == "__main__":
    run()
