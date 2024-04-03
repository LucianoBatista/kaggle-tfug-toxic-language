import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
import polars as pl
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPEN_AI_KEY")

client = OpenAI(
    api_key=OPENAI_API_KEY,
)


def collect_examples():
    df = pl.read_csv("data/train.csv")
    toxic_comments = df.filter(pl.col("label") == 1)
    non_toxic_comments = df.filter(pl.col("label") == 0)

    three_toxic_examples = toxic_comments.sample(3, seed=42)
    three_non_toxic_examples = non_toxic_comments.sample(3, seed=42)

    examples = pl.concat([three_toxic_examples, three_non_toxic_examples])
    return examples


def collect_test_data():
    df = pl.read_csv("data/submission_not_classified.csv")
    print(df.head())
    return df


def call_gpt(examples: pl.DataFrame, new_example):
    examples = examples.sample(fraction=1, seed=2 shuffle=True).to_numpy()

    prompt = """Classifique os comentários abaixo em tóxicos ou não tóxicos, responda com 'Tóxico' ou 'Não Tóxico':

    Comentário: {}
    Classe: {}

    Comentário: {}
    Classe: {}

    Comentário: {}
    Classe: {}

    Comentário: {}
    Classe: {}

    Comentário: {}
    Classe: {}

    Comentário: {}
    Classe: {}

    Comentário: {}
    Classe:""".format(
        examples[0][0],
        "Tóxico" if examples[0][1] == 1 else "Não Tóxico",
        examples[1][0],
        "Tóxico" if examples[1][1] == 1 else "Não Tóxico",
        examples[2][0],
        "Tóxico" if examples[2][1] == 1 else "Não Tóxico",
        examples[3][0],
        "Tóxico" if examples[3][1] == 1 else "Não Tóxico",
        examples[4][0],
        "Tóxico" if examples[4][1] == 1 else "Não Tóxico",
        examples[5][0],
        "Tóxico" if examples[5][1] == 1 else "Não Tóxico",
        new_example,
    )

    messages = [
        {"role": "system", "content": "Você é um assistente muito útil"},
        {"role": "user", "content": prompt},
    ]

    response = client.chat.completions.create(
        messages=messages,
        model="gpt-4-turbo-preview",
    )

    message = response.choices[0].message.content
    return message


if __name__ == "__main__":
    example = collect_examples()
    test_data = collect_test_data()
    test_data = test_data.to_dicts()

    classifications = []
    ids = []
    for item in tqdm(test_data):
        classification = call_gpt(example, item["text"])
        print(classification)
        classifications.append(classification)
        ids.append(item["id"])

    df = pl.DataFrame(
        {
            "id": ids,
            "classification": classifications,
        }
    )

    df.write_csv("data/the_rest.csv")
