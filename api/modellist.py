

import openai

# MODEL = "text-davinci-003"
MODEL = "text-embedding-ada-002"

openai.api_key = "sk-TiEAWDcxuOdvorJkF9394dA95b124d04Af5543090564FbE2"
openai.api_base = "https://api.rcouyi.com/v1"


for d in openai.Model.list().data:
    print(d.id)