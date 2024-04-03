from numpy import mean
import polars as pl
from dataclasses import dataclass
import re
import string
from unidecode import unidecode
import nltk

import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from lightgbm import LGBMClassifier

nltk.download("wordnet")
nltk.download("rslp")


@dataclass
class Config:
    data_train: str = "data/train.csv"
    data_test: str = "data/test.csv"
    tokenizer_path: str = "model/tokenizer"


class Exp07:
    def __init__(self, config: Config):
        self.train = pl.read_csv(config.data_train)
        self.test = pl.read_csv(config.data_test)
        self.stopwords = self._get_stopwords()
        self.clean_train = self._clean_data(self.train)
        self.tfidf = None
        self.clf = None
        self.predictions = None

    def _get_stopwords(self):
        stopwords = list(
            set([unidecode(w) for w in nltk.corpus.stopwords.words("portuguese")])
        )
        stopwords.extend(
            list(set([unidecode(w) for w in nltk.corpus.stopwords.words("english")]))
        )
        stopwords.extend(["rt", "@user", "user"])
        return stopwords

    def _clean_data(self, df: pl.DataFrame, train: bool = True):
        """Remove hashtaghs, numbers, punctuation, accents, links and stopwords."""
        print(df.select("text"))
        if train:
            df = df.drop_nulls("label").unique("text")

        df = df.to_pandas()
        df["text"] = df["text"].apply(
            lambda x: re.sub("#[^ ]+", "", x)
        )  # remove hashtags

        df["text"] = df["text"].apply(lambda x: re.sub(r"\d+", "", x))  # remove numbers

        df["text"] = df["text"].apply(
            lambda x: x.translate(str.maketrans("", "", string.punctuation))
        )  # remove punctuation

        df["text"] = df["text"].apply(lambda x: unidecode(x))  # remove accents

        df["text"] = df["text"].apply(
            lambda x: re.sub("http[^ ]+", "", x)
        )  # remove links

        df["text"] = df["text"].apply(
            lambda x: " ".join(w.strip() for w in x.split() if w not in self.stopwords)
        )  # remove stopword

        # stemming
        stemmer = nltk.stem.RSLPStemmer()
        df["text"] = df["text"].apply(
            lambda x: " ".join([stemmer.stem(w) for w in x.split()])
        )

        df = pl.from_pandas(df)
        print(df.select("text"))
        return df

    def split_data(
        self, df: pl.DataFrame
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """
        Split data into train and test
        """
        X_train, X_test, y_train, y_test = train_test_split(
            df.select("text"),
            df.select("label"),
            test_size=0.2,
            random_state=42,
            stratify=df.select("label"),
        )
        return X_train, X_test, y_train, y_test

    def train_tfidf(self, train_tokenized: list[str]):
        """
        Train tfidf vectorizer
        """
        self.tfidf = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
        )
        self.tfidf.fit(train_tokenized)

    def vectorize(self, train_tokenized: list[str]):
        return self.tfidf.transform(train_tokenized)

    def exp12_model(self, x_train, y_train):
        """
        Train and evaluate model
        """
        clf = MultinomialNB(alpha=0.02)
        sgd_model = SGDClassifier(max_iter=8000, tol=1e-4, loss="modified_huber")
        p6 = {
            "n_iter": 15000,
            "verbose": -1,
            "objective": "binary",
            "metric": "auc",
            "learning_rate": 0.05073909898961407,
            "colsample_bytree": 0.726023996436955,
            "colsample_bynode": 0.5803681307354022,
            "lambda_l1": 8.562963348932286,
            "lambda_l2": 4.893256185259296,
            "min_data_in_leaf": 115,
            "max_depth": 23,
            "max_bin": 898,
        }
        lgb = LGBMClassifier(**p6)
        lr = LogisticRegression(max_iter=3000, random_state=42, n_jobs=-1)

        # Creating the ensemble model
        # ensemble = VotingClassifier(
        #     estimators=[("sgd", sgd_model), ("lr", lr), ("lgb", lgb), ("nb", clf)],
        #     weights=[0.2, 0.5, 0.3, 0.1],
        #     voting="soft",
        #     n_jobs=-1,
        # )
        #
        # # Fit the ensemble model
        # ensemble.fit(x_train, y_train)
        model = lr.fit(x_train, y_train)
        # hist_clf = HistGradientBoostingClassifier()
        # model = hist_clf.fit(x_train.toarray(), y_train)
        self.clf = model

    #
    # def submission(self, real_test):
    #     """
    #     Generate submission file
    #     """
    #     real_test_text = real_test.select("text").to_numpy().flatten().tolist()
    #
    #     real_test_tokenized = []
    #     for text in tqdm(real_test_text):
    #         text_tokenized = self.tokenizer.tokenize(text)
    #         real_test_tokenized.append(" ".join(text_tokenized))
    #
    #     real_test_transformed = self.tfidf.transform(real_test_tokenized)
    #     predictions = self.clf.predict_proba(real_test_transformed)[:, 1]
    #     df = pl.DataFrame(
    #         {
    #             "id": real_test.select("id").to_numpy().flatten().tolist(),
    #             "generated": predictions,
    #         }
    #     )
    #     df.write_csv("submission_tfidf.csv")
    #
    def evaluate_model(self, X_test: list[str], y_test: list[str]):
        """
        Evaluate model
        """
        predictions = self.clf.predict_proba(X_test)[:, 1]
        threshold = 0.5

        predictions = [1 if p > threshold else 0 for p in predictions]
        acc = mean([1 if p == y else 0 for p, y in zip(predictions, y_test)])
        print(acc)
        # predictions = self.clf.predict(X_test)
        # print(f"ACC score: {self.clf.score(X_test, y_test)}")
        # print(f"ROC AUC score: {roc_auc_score(y_test, predictions)}")


if __name__ == "__main__":
    config = Config()
    exp07 = Exp07(config)
    X_train, X_test, y_train, y_test = exp07.split_data(exp07.clean_train)

    # X_train_tokenized = X_train.to_numpy().flatten().tolist()
    # X_test_tokenized = X_test.to_numpy().flatten().tolist()
    exp07.train_tfidf(X_train.to_numpy().flatten().tolist())
    X_train = exp07.vectorize(X_train.to_numpy().flatten().tolist())
    X_test = exp07.vectorize(X_test.to_numpy().flatten().tolist())
    # exp06.exp06_w_cv()
    exp07.exp12_model(X_train, y_train)
    exp07.evaluate_model(X_test, y_test.to_numpy().flatten().tolist())

    # if not KAGGLE:
    #     exp06.submission(data_preprocessing.data_test)
