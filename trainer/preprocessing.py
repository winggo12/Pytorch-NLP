## This file has the following funcstions
## 1. get_dataset(df_train, df_val)
## By inputting the dataframe of train and validation
## Return dataset class that include dataloader of train and validation

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer

from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

import pandas as pd
import os
import sys

root_path = os.path.abspath(os.getcwd())
path_list = root_path.split("/")
index = path_list.index('Pytorch-NLP')
parent_path = ""
for i in range(index + 1): parent_path += (path_list[i] + "/")
# print("Parent Directory: "+parent_path)
sys.path.append(parent_path)

class Dataset():
    def __init__(self, dataset_type, data_frame, attention_masks, batch_size=16):
        self.dataset_type = dataset_type
        self.data_frame = data_frame
        self.attention_masks = attention_masks
        self.batch_size = batch_size
        self.data_loader = None


def get_dataset(df_train, df_val):
    attention_masks = []
    train_dataset = Dataset("train", df_train, attention_masks)
    val_dataset = Dataset("val", df_val, attention_masks)

    for dataset in [train_dataset, val_dataset]:
        df = dataset.data_frame
        print("Handling dataset: ", dataset.dataset_type)
        df.rename(columns={'label': 'label_name'}, inplace=True)
        df['label'] = labelencoder.fit_transform(df['label_name'])
        # print(df)

        sentences = df.sentence.values
        # print("Distribution of data based on labels: ", df.label.value_counts())
        MAX_LEN = 256
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        tokenized_inputs = []
        for sentence in sentences:
            tokenized_inputs.append(
                tokenizer.encode(sentence, add_special_tokens=True,
                                 max_length=MAX_LEN, pad_to_max_length=True,
                                 truncation=True))

        labels = df.label.values

        # print("Actual sentence before tokenization: ", sentences[2])
        # print("Encoded Input from dataset: ", tokenized_inputs[2])

        ## Create attention mask
        attention_masks = []
        for input in tokenized_inputs:
            attention_mask = []
            for element in input:
                if element > 0:
                    attention_mask.append(float(1))
                else:
                    attention_mask.append(float(0))
            attention_masks.append(attention_mask)

        # attention_masks = [[float(i > 0) for i in seq] for seq in tokenized_inputs]
        attention_masks = torch.tensor(attention_masks)
        dataset.attention_masks = attention_masks
        inputs = torch.tensor(tokenized_inputs)
        labels = torch.tensor(labels)

        tensor_dataset = TensorDataset(inputs, attention_masks, labels)
        random_sampler = RandomSampler(tensor_dataset)
        data_loader = DataLoader(tensor_dataset, sampler=random_sampler, batch_size=dataset.batch_size)
        dataset.data_loader = data_loader

    return train_dataset, val_dataset, tokenizer


if __name__ == '__main__':
    df_train = pd.read_csv(parent_path + "data/train.txt",
                           delimiter=';', names=['sentence', 'label'])
    # df_test = pd.read_csv(parent_path + "data/test_data.txt",
    #                       delimiter=';', names=['sentence', 'label'])
    df_val = pd.read_csv(parent_path + "data/val.txt",
                         delimiter=';', names=['sentence', 'label'])

    train_dataset, val_dataset, tokenizer = get_dataset(df_train, df_val)
