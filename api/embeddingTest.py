import openai
from openai.embeddings_utils import cosine_similarity, get_embedding

# MODEL = "text-davinci-003"
MODEL = "text-embedding-ada-002"

openai.api_key = "sk-TiEAWDcxuOdvorJkF9394dA95b124d04Af5543090564FbE2"
openai.api_base = "https://api.rcouyi.com/v1"


# 获取"好评"和"差评"的
positive_review = get_embedding("好评",engine=MODEL)
negative_review = get_embedding("差评,体验差,商品质量差,价格高",engine=MODEL)

positive_example = get_embedding("真不错",engine=MODEL)
negative_example = get_embedding("价格高",engine=MODEL)

def get_score(sample_embedding):
  return cosine_similarity(sample_embedding, positive_review) - cosine_similarity(sample_embedding,negative_review)

positive_score = get_score(positive_example)
negative_score = get_score(negative_example)

print("好评例子的评分 : %f" % (positive_score))
print("差评例子的评分 : %f" % (negative_score))