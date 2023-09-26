import csv

import numpy as np
import pandas as pd
import torch
from transformers import T5Tokenizer, T5Model

MAX_INDEX = 600000

EMBEDDING = 'embedding'

datafile_path = "F:\\ai\\dataset\\Reviews.csv"
df = pd.read_csv(datafile_path)

print(len(df))
# row = df.head(1)
# print("评价：", row["Summary"][0], "评分",row["Score"][0],"embed",row[EMBEDDING])

tokenizers = T5Tokenizer.from_pretrained("t5-small",model_max_length=512)
model = T5Model.from_pretrained('t5-small')
model.eval()


def get_t5_vector(line):
    input_ids = tokenizers.encode(line,return_tensors='pt',max_length=512,truncation="only_first")
    with torch.no_grad():
        outputs = model.encoder(input_ids=input_ids)
        vector = outputs.last_hidden_state.mean(dim=1)
    return vector[0]
# del df[EMBEDDING]
# df.insert(len(df.columns),EMBEDDING,None)
# df[EMBEDDING] = df[EMBEDDING].astype(str)


for index,row in df.iterrows():
    if index >= MAX_INDEX:
        break
    if row[EMBEDDING] is not None and type(row[EMBEDDING]) != float or not np.isnan(row[EMBEDDING]):
        continue
    summary_ = row["Summary"]
    if type(summary_) == float:
        print("发现浮点数在评论区",summary_)
        df.drop(index, inplace=True)
        continue
    vector = get_t5_vector(summary_)
    # df.loc[index,EMBEDDING] = np.array2string(vector, separator=",")
    df.loc[index,EMBEDDING] = np.array2string(vector.numpy(),separator=",").replace("[","").replace("]","")
    if index % 200 == 0:
        print("当前处理行数为",index)

df.to_csv(datafile_path,index=False)

print("完成")