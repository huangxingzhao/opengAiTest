import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from transformers import T5Tokenizer, T5Model
from numpy import vectorize
import numpy as np

MAX_INDEX = 1000

EMBEDDING = 'embedding'

datafile_path = "F:\\ai\\dataset\\Reviews.csv"
df = pd.read_csv(datafile_path)
print(len(df))
# row = df.head(1)
# print("评价：", row["Summary"][0], "评分",row["Score"][0],"embed",row[EMBEDDING])

tokenizers = T5Tokenizer.from_pretrained("t5-small",model_max_length=512)
model = T5Model.from_pretrained('t5-small')
model.eval()

def str_to_array(s):
    return np.fromstring(s, sep=",", dtype=float)


testRows = df
embeddings = testRows["embedding"]
myList = []
for row in embeddings:
    myList.append(str_to_array(row))

scores = testRows["Score"]

X_train, X_test, y_train, y_test = train_test_split(
    myList, scores, test_size=0.2, random_state=42
)


clf = LogisticRegression()
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
probas = clf.predict_proba(X_test)

report = classification_report(y_test, preds)
print(report)


print("完成")