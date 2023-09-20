import openai

# MODEL = "text-davinci-003"
MODEL = "text-davinci-003"

openai.api_key = "sk-TiEAWDcxuOdvorJkF9394dA95b124d04Af5543090564FbE2"
openai.api_base = "https://api.rcouyi.com/v1"

prompt = """
Consideration product : 工厂现货PVC充气青蛙夜市地摊热卖充气玩具发光蛙儿童水上玩具

 1. Compose human readable product title used on Amazon in english within 20 words
 2. Write 5 selling points for the products in Amazon.
 3. Evaluate a price range for this product in U.S.

 Output the result in json format with three properties called title, selling_poin
 """

def get_response(prompt):

    completions = openai.Completion.create(model=MODEL, prompt=prompt, max_tokens=100, n=1, stop=None, temperature=0.0)
    message = completions.choices[0].text
    return message

print(get_response(prompt))


