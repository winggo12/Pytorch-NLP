import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn.functional as F
from transformers import BertTokenizer, BertConfig,AdamW, BertForSequenceClassification,get_linear_schedule_with_warmup


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def load_model_tokenizer(model_path, tokenizer_path):
    model = torch.load(model_path)
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    return model, tokenizer

def inference(model, tokenizer, input):
    MAX_LEN = 256

    encoded_input = [tokenizer.encode(sent, add_special_tokens=True, max_length=MAX_LEN, pad_to_max_length=True) for sent in
                 input]
    attention_masks = []
    for list in encoded_input:
        attention_mask = []
        for element in list:
            if element > 0 : attention_mask.append(float(1))
            else: attention_mask.append(float(0))
        attention_masks.append(attention_mask)

    print("Actual sentence before tokenization: ", input[0])
    print("Encoded Input from dataset: ", encoded_input[0])
    print("Attention Mask: ", attention_masks[0])

    np_encoded_input = np.asarray(encoded_input)
    np_mask = np.asarray(attention_masks)
    tensor_encoded_input = torch.from_numpy(np_encoded_input)
    tensor_mask = torch.from_numpy(np_mask)
    # with torch.no_grad:
    print("Inference Completed")
    return 0


if __name__ == '__main__':
    input = ["i am gay"]
    model, tokenizer = load_model_tokenizer(model_path="../data_preprocessing/model/pytorch_model.bin",
                         tokenizer_path="../data_preprocessing/tokenizer/")
    inference(model, tokenizer, input)
