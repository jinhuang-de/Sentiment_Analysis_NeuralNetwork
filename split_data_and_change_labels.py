#!/usr/bin/env python
# coding: utf-8

# In[27]:


import pandas as pd
from numpy.random import RandomState

rng = RandomState()
df = pd.read_csv('./data/sentiment140_processed.csv', sep='\t')

train = df.sample(frac=0.7, random_state=rng)

test_and_valid = df.loc[~df.index.isin(train.index)]

test = test_and_valid.sample(frac=0.5, random_state=rng)
valid = test_and_valid.loc[~test_and_valid.index.isin(test.index)]

docs = ["train", "test", "valid"]

train.to_csv(r'./data/train.csv', sep='\t', encoding='utf-8', index=False)
test.to_csv(r'./data/test.csv', sep='\t', encoding='utf-8', index=False)
valid.to_csv(r'./data/valid.csv', sep='\t', encoding='utf-8', index=False)

for d in docs:
    print(d,':')
    path = './data/'+ d + '.csv'
    d = pd.read_csv(path, sep = '\t')
    d["label"].replace({4: 1}, inplace=True)
    d.to_csv(path, sep='\t', encoding='utf-8', index=False)
    print(d.groupby('label').count(), '\n') 


# In[ ]:




