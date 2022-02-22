import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn.functional as F
from transformers import BertTokenizer, BertConfig, BertModel, AdamW, BertForSequenceClassification,get_linear_schedule_with_warmup


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

labels = {
    0: "anger",
    1: "fear",
    2: "joy",
    3: "love",
    4: "sadness",
    5: "surprise"
}

def load_model_tokenizer( model_path, tokenizer_path):
    model =BertForSequenceClassification.from_pretrained(model_path)
    model.eval()
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

    print("Inference Completed")
    print("tensor_encoded_input: ", tensor_encoded_input)
    print("tensor_mask", tensor_mask)

    with torch.no_grad():
        Seq_Classifier_Out = model(tensor_encoded_input, token_type_ids=None, attention_mask=tensor_mask)

    logits = Seq_Classifier_Out['logits']
    print(logits)

    logits = logits.detach().numpy()
    pred_flat = np.argmax(logits, axis=1).flatten()
    print(pred_flat)
    return pred_flat[0]

def emotion(pred ,label_dict):
    return label_dict[pred]


if __name__ == '__main__':
    input = ["i am happy"]
    model, tokenizer = load_model_tokenizer(model_path="../data_preprocessing/model",
                                            tokenizer_path="../data_preprocessing/tokenizer/")
    pred = inference(model, tokenizer, input)
    emotion = emotion(pred, labels)
    print(emotion)

