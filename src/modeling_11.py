import polars as pl
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
import os
from tenacity import retry, stop_after_attempt, wait_random_exponential


load_dotenv()

OPENAI_API_KEY = os.getenv("OPEN_AI_KEY")

client = OpenAI(
    api_key=OPENAI_API_KEY,
)


def collect_examples():
    df = pl.read_csv("data/train.csv")
    toxic_comments = df.filter(pl.col("label") == 1)
    non_toxic_comments = df.filter(pl.col("label") == 0)

    three_toxic_examples = toxic_comments.sample(5, seed=2)
    three_non_toxic_examples = non_toxic_comments.sample(5, seed=2)

    examples = pl.concat([three_toxic_examples, three_non_toxic_examples])
    return examples


def collect_test_data():
    df = pl.read_csv("data/test.csv")
    print(df.head())
    return df


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def call_gpt(new_example) -> str:

    messages = [
        {
            "role": "system",
            "content": "Haja como um moderador que detecta conteúdo tóxico, onde tóxico é representado por 1, e não tóxico por 0.",
        },
        {"role": "user", "content": new_example},
    ]

    response = client.chat.completions.create(
        messages=messages,
        model="ft:gpt-3.5-turbo-0125:oc3-academy:kaggle:98qoVkQt",
        top_p=1,
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
        classification = call_gpt(item["text"])
        print(classification)
        classifications.append(classification)
        print(item)
        ids.append(item["text"])

    df = pl.DataFrame(
        {
            "text": ids,
            "classification": classifications,
        }
    )

    df.write_csv("data/test_tuned_gpt.csv")
