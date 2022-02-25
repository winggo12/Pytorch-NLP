import torch
from transformers import BertTokenizer, BertConfig,AdamW, BertForSequenceClassification,get_linear_schedule_with_warmup

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from preprocessing import get_data_loader

from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

from config import config
from utils.utils import progress_bar, plot_final_acc_loss, classification_report_csv


def train(train_dataset, val_dataset):
    learning_rate = config.learning_rate
    adam_epsilon = config.adam_epsilon
    num_of_epoch = config.num_of_epoch
    num_labels = 6
    num_training_steps = len(train_dataset.data_loader) * num_of_epoch

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels).to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=adam_epsilon, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    model.zero_grad()

    train_val_acc_loss_dict = {'train':{'acc': [], 'loss':[]},
                               'val': {'acc': [], 'loss':[]} }

    for epoch in range(1, num_of_epoch + 1):
        print("Epoch Number: ", epoch)
        accumulated_loss = {'train':0, 'val':0}
        reporting_iteration = {'train':0, 'val':0 }
        running_iteration = {'train':0, 'val':0 }
        running_prediction = {'train': [], 'val': []}
        running_label = {'train': [], 'val': []}
        # running_loss = {'train':0, 'val':0 }
        # running_num_of_data = {'train':0, 'val':0}

        for dataset in [train_dataset, val_dataset]:
            dataset_type = dataset.dataset_type
            data_loader = dataset.data_loader
            reporting_iteration[dataset_type] = int(len(data_loader)/50)
            print(f"[ {dataset_type} dataset]")
            for step, batch in enumerate(data_loader):
                progress_bar(current=running_iteration[dataset_type],
                             total=len(data_loader))
                if dataset.dataset_type == 'train' :  model.train()
                elif dataset.dataset_type == 'val' :  model.eval()

                batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
                outputs = None
                if dataset.dataset_type == 'train':
                    outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
                    predictions = outputs['logits'].to('cpu').detach().numpy()
                    label_ids = b_labels.to('cpu').detach().numpy()

                    loss = outputs[0]
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                elif dataset.dataset_type == 'val':
                    with torch.no_grad():
                        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
                        predictions = outputs[0].to('cpu').numpy()
                        label_ids = b_labels.to('cpu').numpy()

                flatten_prediction = np.argmax(predictions, axis=1).flatten()
                flatten_label = label_ids.flatten()

                running_iteration[dataset_type] += 1
                running_prediction[dataset_type].extend(flatten_prediction)
                running_label[dataset_type].extend(flatten_label)
                accumulated_loss[dataset_type] += loss.item()

            final_acc = accuracy_score(
                running_label[dataset_type],
                running_prediction[dataset_type])
            final_loss = accumulated_loss[dataset_type] / len(dataset.data_loader)

            train_val_acc_loss_dict[dataset_type]['acc'].append(final_acc)
            train_val_acc_loss_dict[dataset_type]['loss'].append(final_loss)

            print(f"(dataset_type) acc: {final_acc}, loss: {final_loss}")

            if epoch == num_of_epoch :
                cr = classification_report(running_label[dataset_type], running_prediction[dataset_type],
                                           output_dict=True)
                df = pd.DataFrame(cr).transpose()
                df.to_csv(f"./{dataset_type}_classification_report.csv")
                # print(cr)
                cm = confusion_matrix(running_label[dataset_type], running_prediction[dataset_type])
                # print(cm)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                disp.plot()
                plt.savefig(f"{dataset_type}_ConfusionMatrix.jpg")
                plt.close()

    y(train_val_acc_loss_dict)

    return model

def save_model(model, tokenizer):
    model_save_folder = 'model/'
    tokenizer_save_folder = 'tokenizer/'
    path_model = './' + model_save_folder
    path_tokenizer = './' + tokenizer_save_folder

    model.save_pretrained(path_model)
    tokenizer.save_pretrained(path_tokenizer)

    model_save_name = 'fineTuneModel.pt'
    path = path_model = './' + model_save_folder + '/' + model_save_name
    torch.save(model.state_dict(), path)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df_train = pd.read_csv("../data/train.txt",
                           delimiter=';', names=['sentence', 'label'])
    df_test = pd.read_csv("../data/test_data.txt",
                          delimiter=';', names=['sentence', 'label'])
    df_val = pd.read_csv("../data/val.txt",
                         delimiter=';', names=['sentence', 'label'])
    train_dataset, val_dataset, tokenizer = get_data_loader(df_train, df_val)
    model = train(train_dataset, val_dataset)
    save_model(model, tokenizer)