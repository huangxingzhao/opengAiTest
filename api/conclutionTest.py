from sklearn.datasets import fetch_20newsgroups
import pandas as pd
# todo
# newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'foo
#
# df = pd.DataFrame([newsgroups_train.data, newsgroups_train.target.tolist()]).
# df.columns = ['text', 'target']
# targets = pd.DataFrame( newsgroups_train.target_names, columns=['title'])
#
# out = pd.merge(df, targets, left_on='target', right_index=True)
# out.to_csv('20_newsgroup.csv', index=False)