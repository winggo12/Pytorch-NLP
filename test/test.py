##This file read the test_data and write the test_prediction.txt

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

from inference.inference import inference,batch_inference,  emotion, load_model_tokenizer

if __name__ == '__main__':
    labels = {
        0: "anger",
        1: "fear",
        2: "joy",
        3: "love",
        4: "sadness",
        5: "surprise"
    }
    model, tokenizer = load_model_tokenizer(model_path=parent_path + "trainer/model",
                                            tokenizer_path=parent_path + "trainer/tokenizer/")

    df_test = pd.read_csv(parent_path+ "data/test_data.txt",
                          delimiter=';', names=['sentence', 'label'])
    # df_test = df_test[:20]
    inputs = []
    with open(parent_path+'test/test_prediction.txt', 'w') as f, open(parent_path+'test/trial.txt', 'w') as trial:
        for index, row in df_test.iterrows():
            input = row['sentence']
            inputs.append(input)
            if len(inputs) == 400 or index == ( len(df_test) - 1 ):
                preds = batch_inference(model, tokenizer, inputs)
                feelings = []
                for pred in preds:
                    feeling = emotion(pred, labels)
                    feelings.append(feeling)
                    f.write(str(feeling)+ "\n")

                for i in range(len(feelings)):
                    comment = inputs[i], " ", feelings[i]
                    print(comment)
                    trial.write(str(comment)+ "\n")

                inputs = []
        trial.close()
        f.close()