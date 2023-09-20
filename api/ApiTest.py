import openai
import os

# MODEL = "text-davinci-003"
MODEL = "text-davinci-003"

openai.api_key = "sk-TiEAWDcxuOdvorJkF9394dA95b124d04Af5543090564FbE2"
openai.api_base = "https://api.rcouyi.com/v1"


prompt = """如何制作番茄炒蛋"""

def get_response(prompt):

    completions = openai.Completion.create(model=MODEL, prompt=prompt, max_tokens=100, n=1, stop=None, temperature=0.0)
    message = completions.choices[0].text
    return message

print(get_response(prompt))


