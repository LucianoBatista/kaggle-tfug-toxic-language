from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential
import os
from dotenv import load_dotenv
import rich


load_dotenv()

OPENAI_API_KEY = os.getenv("OPEN_AI_KEY")

client = OpenAI(
    api_key=OPENAI_API_KEY,
)


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def moderate(input):
    response = client.moderations.create(input=input)
    return response.results[0]


if __name__ == "__main__":
    input = "rt @user deus só não me fez magra pq sabe q se fizesse eu iria ser uma grande vadia"
    rich.print(moderate(input))
