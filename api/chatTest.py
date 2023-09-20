import openai

MODEL = "text-davinci-003"

openai.api_key = "sk-TiEAWDcxuOdvorJkF9394dA95b124d04Af5543090564FbE2"
openai.api_base = "https://api.rcouyi.com/v1"


prompt = '请你用朋友的语气回复给到客户，并称他为“亲”，他的订单已经发货在路上了，预计在3天之内会飞到他的手中,注意只能使用中文回复'

def get_response(prompt,temperature = 1.0):
    completions = openai.Completion.create(model=MODEL, prompt=prompt, max_tokens=400, n=1, stop=None,
                                      temperature=temperature)

    return completions.choices[0].text

print(get_response(prompt))
