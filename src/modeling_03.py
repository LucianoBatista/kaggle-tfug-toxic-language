import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
import polars as pl
from tqdm import tqdm


def collect_examples():
    df = pl.read_csv("data/train.csv")
    toxic_comments = df.filter(pl.col("label") == 1)
    non_toxic_comments = df.filter(pl.col("label") == 0)

    three_toxic_examples = toxic_comments.sample(3, seed=42)
    three_non_toxic_examples = non_toxic_comments.sample(3, seed=42)

    examples = pl.concat([three_toxic_examples, three_non_toxic_examples])
    return examples


def collect_test_data():
    df = pl.read_csv("data/test.csv")
    print(df.head())
    return df


def trying_sabia(examples: pl.DataFrame, new_example, model, tokenizer):
    examples = examples.sample(fraction=1, seed=42, shuffle=True).to_numpy()

    prompt = """Classifique os comentários abaixo em tóxicos ou não tóxicos:

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

    print(prompt)

    input_ids = tokenizer(prompt, return_tensors="pt")

    output = model.generate(
        input_ids["input_ids"].to("cuda"),
        max_length=1024,
        eos_token_id=tokenizer.encode("\n"),
    )  # Stop generation when a "\n" token is dectected

    # The output contains the input tokens, so we have to skip them.
    output = output[0][len(input_ids["input_ids"][0]) :]

    return tokenizer.decode(output, skip_special_tokens=True)


if __name__ == "__main__":
    example = collect_examples()
    test_data = collect_test_data()
    test_data = test_data.to_dicts()

    tokenizer = LlamaTokenizer.from_pretrained("maritaca-ai/sabia-7b")
    model = LlamaForCausalLM.from_pretrained(
        "maritaca-ai/sabia-7b",
        device_map="auto",  # Automatically loads the model in the GPU, if there is one. Requires pip install acelerate
        torch_dtype=torch.bfloat16,  # If your GPU does not support bfloat16, change to torch.float16
    )

    classifications = []
    ids = []
    for item in tqdm(test_data):
        classification = trying_sabia(example, item["text"], model, tokenizer)
        classifications.append(classification)
        ids.append(item["id"])

    df = pl.DataFrame(
        {
            "id": ids,
            "classification": classifications,
        }
    )

    df.write_csv("data/submission_inter.csv")
