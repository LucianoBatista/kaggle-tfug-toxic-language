from collections import Counter
import polars as pl
from nltk.corpus import stopwords
import nltk


nltk.download("stopwords")


def read_data(file_path: str) -> pl.DataFrame:
    return pl.read_csv(file_path)


if __name__ == "__main__":
    df = read_data("data/train.csv")
    print(df.head())

    # duplicated rows
    print(df.select(pl.col("text").value_counts(sort=True)).unnest("text"))

    # destribution of target
    print(df.select(pl.col("label").value_counts(sort=True)).unnest("label"))

    # missing values
    # we have missings on the label and also on the text
    print(df.filter(pl.col("text").is_null()))
    print(df.filter(pl.col("label").is_null()))

    # words count for toxic
    texts = df.filter(pl.col("label") == 1).select("text").to_numpy().flatten().tolist()
    texts_wo_stopwords = [
        word
        for word in " ".join(texts).split()
        if word not in stopwords.words("portuguese")
    ]
    freq_words = Counter(texts_wo_stopwords).most_common(100)
    less_freq_words = Counter(texts_wo_stopwords).most_common()[: -100 - 1 : -1]

    # words count for non-toxic
    texts = df.filter(pl.col("label") == 0).select("text").to_numpy().flatten().tolist()
    texts_wo_stopwords = [
        word
        for word in " ".join(texts).split()
        if word not in stopwords.words("portuguese")
    ]
    freq_words = Counter(texts_wo_stopwords).most_common(100)
    less_freq_words = Counter(texts_wo_stopwords).most_common()[: -100 - 1 : -1]
