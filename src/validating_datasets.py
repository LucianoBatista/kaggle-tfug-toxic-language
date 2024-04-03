from datasets import load_dataset
from datasets import DatasetDict, Dataset
import polars as pl
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    emotions = load_dataset("emotion")
    print(emotions)

    df = pl.read_csv("data/train.csv").to_pandas()

    train, test = train_test_split(df, test_size=0.2, random_state=42)

    train_dataset = Dataset.from_pandas(train)
    test_dataset = Dataset.from_pandas(test)
    print(DatasetDict({"train": train_dataset, "test": test_dataset}))
