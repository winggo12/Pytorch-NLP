import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn.functional as F
from transformers import BertTokenizer, BertConfig,AdamW, BertForSequenceClassification,get_linear_schedule_with_warmup


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
# Import and evaluate each test batch using Matthew's correlation coefficient
from sklearn.metrics import accuracy_score,matthews_corrcoef

from tqdm import tqdm, trange,tnrange,tqdm_notebook
import random
import os
import io

df_train = pd.read_csv("../data/train.txt",
                       delimiter=';', names=['sentence','label'])
df_test = pd.read_csv("../data/test_data.txt",
                      delimiter=';', names=['sentence','label'])
df_val = pd.read_csv("../data/val.txt",
                     delimiter=';', names=['sentence','label'])

df = pd.concat([df_train,df_val])

labels = df['label'].unique()

print(labels)

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
df['label_enc'] = labelencoder.fit_transform(df['label'])

print(df['label_enc'].head())
print(df['label'].head())

df[['label','label_enc']].drop_duplicates(keep='first')

print(df.head())

df.rename(columns={'label':'label_desc'},inplace=True)
df.rename(columns={'label_enc':'label'},inplace=True)


print(df.head())from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
df['label_enc'] = labelencoder.fit_transform(df['label'])

print(df['label_enc'].head())
print(df['label'].head())

df[['label','label_enc']].drop_duplicates(keep='first')

print(df.head())

df.rename(columns={'label':'label_desc'},inplace=True)
df.rename(columns={'label_enc':'label'},inplace=True)


print(df.head())


