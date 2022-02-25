import pandas as pd
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
    model, tokenizer = load_model_tokenizer(model_path="../trainer/model",
                                            tokenizer_path="../trainer/tokenizer/")

    df_test = pd.read_csv("../data/test_data.txt",
                          delimiter=';', names=['sentence', 'label'])
    inputs = []
    with open('test_prediction.txt', 'w') as f, open('trial.txt', 'w') as trial:
        for index, row in df_test.iterrows():
            input = row['sentence']
            inputs.append(input)
            if len(inputs) == 400:
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