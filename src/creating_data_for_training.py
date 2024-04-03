import polars as pl
import json


def create_json_to_fine_tuning_jog():
    df = pl.read_csv("data/train.csv")
    df = df.drop_nulls("label").unique("text").sample(n=50, seed=42)

    df_dicts = df.to_dicts()

    jsons = []

    for item in df_dicts:
        json = {
            "messages": [
                {
                    "role": "system",
                    "content": "Haja como um moderador que detecta conteúdo tóxico, onde tóxico é representado por 1, e não tóxico por 0.",
                },
                {"role": "user", "content": item["text"]},
                {
                    "role": "assistant",
                    "content": str(item["label"]),
                },
            ]
        }
        jsons.append(json)

    return jsons


if __name__ == "__main__":
    jsons_train_job = create_json_to_fine_tuning_jog()
    print(jsons_train_job)
    # create a jsonl file
    with open("data/train_job.jsonl", "w") as f:
        for i in jsons_train_job:
            data = json.dumps(i, ensure_ascii=False)
            f.write(data + "\n")
