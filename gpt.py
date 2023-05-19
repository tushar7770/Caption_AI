import os
import openai
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv('api_key')


def get_captions(description):
    prompt = "create 10 catchy caption from the description of image try to avoid objectify them. Description: " + \
        str(description)
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return str(response["choices"][0]["text"])
