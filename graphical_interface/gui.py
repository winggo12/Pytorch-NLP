## This file has the following class
## 1. Window(QMainWindow)
## The main GUI is defined in this class, run this python file
## to open the GUI.


from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtGui
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import os
import sys

root_path = os.path.abspath(os.getcwd())
print("Current Working Directory: ", root_path)
sys.path.append(root_path)
from inference.inference import load_model_tokenizer, inference, emotion

class Window(QMainWindow):


    def __init__(self):
        super().__init__()

        self.model, self.tokenizer, self.labels = self.init_nlp()
        self.setWindowTitle("Emotional Analysis of Text")
        self.setGeometry(100, 100, 480, 480)
        self.UiComponents()
        self.show()

    def init_nlp(self):
        device_name = 'cpu'
        model, tokenizer = load_model_tokenizer(model_path="../trainer/model",
                                                tokenizer_path="../trainer/tokenizer")
        labels = {
            0: "anger",
            1: "fear",
            2: "joy",
            3: "love",
            4: "sadness",
            5: "surprise"
        }
        return model, tokenizer, labels


    def UiComponents(self):
        sentence_input = QLineEdit("Enter you Sentence in here: (e.g. I am happy)", self)
        sentence_input.setGeometry(80, 40, 300, 40)
        emotion_label = QLabel("Sentence's Emotion will be displayed in here", self)

        emotion_label.setGeometry(80, 80, 300, 60)
        emotion_label.setWordWrap(True)

        img_label = QLabel(self)
        img_label.setGeometry(80, 140, 300, 300)
        pixmap = QPixmap('./image/emotion.jpg')
        pixmap = pixmap.scaled(300, 300, QtCore.Qt.KeepAspectRatio)
        img_label.setPixmap(pixmap)
        sentence_input.returnPressed.connect(lambda: do_action())

        def do_action():
            value = sentence_input.text()
            input = [str(value)]
            pred = inference(self.model, self.tokenizer, input)
            feeling = emotion(pred, self.labels)
            print(feeling)
            emotion_label.setText(str(feeling))
            pixmap = QPixmap('./image/' + str(feeling) + '.jpg')
            pixmap = pixmap.scaled(300, 300, QtCore.Qt.KeepAspectRatio)
            img_label.setPixmap(pixmap)


if __name__ == '__main__':
    # pred = inference(model, tokenizer, input)
    # emotion = emotion(pred, labels)
    # print(emotion)

    App = QApplication(sys.argv)
    window = Window()
    sys.exit(App.exec())