import numpy as np
import openai
import pandas as pd
from openai.embeddings_utils import cosine_similarity, get_embedding
from sklearn.metrics import classification_report
from sklearn.metrics import PrecisionRecallDisplay

# MODEL = "text-davinci-003"
MODEL = "text-embedding-ada-002"

openai.api_key = "sk-TiEAWDcxuOdvorJkF9394dA95b124d04Af5543090564FbE2"
openai.api_base = "https://api.rcouyi.com/v1"

datafile_path = "data/fine_food_reviews_with_embeddings_1k.csv"
df = pd.read_csv(datafile_path)
df["embedding"] = df.embedding.apply(eval).apply(np.array)

df = df[df.Score != 3]
df["sentiment"] = df.Score.replace({1: "negative", 2: "negative", 4: "positive", 5: "positive"})


def evaluate_embeddings_approach(
  labels =['negative', 'positive'],
  model = MODEL,
):
  label_embeddings = [get_embedding(label, engine=model) for label in labels]

  def label_score(review_embedding,label_embeddings):
    return cosine_similarity(review_embedding,label_embeddings[1]) - cosine_similarity(review_embedding,label_embeddings[0])

  probas = df["embedding"].apply(lambda x: label_score(x,label_embeddings))
  preds = probas.apply(lambda x: 'positive' if x > 0 else 'negative')

  report = classification_report(df.sentiment,preds)
  print(report)

  display = PrecisionRecallDisplay.from_predictions(df.setiment,probas,pos_label='abc')

evaluate_embeddings_approach(labels=['An Amazon review with a negative sentiment.','An Amazon review with a positive sentiment.'])