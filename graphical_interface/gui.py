# importing libraries
from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtGui
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import sys
from inference.inference import load_model_tokenizer, inference, emotion

class Window(QMainWindow):


    def __init__(self):
        super().__init__()

        self.model, self.tokenizer, self.labels = self.init_nlp()
        # setting title
        self.setWindowTitle("Emotional Analysis of Text")

        # setting geometry
        self.setGeometry(100, 100, 480, 480)

        # calling method
        self.UiComponents()

        # showing all the widgets
        self.show()

    def init_nlp(self):
        device_name = 'cpu'
        model, tokenizer = load_model_tokenizer(model_path="../data_preprocessing/model/",
                                                tokenizer_path="../data_preprocessing/tokenizer/")
        labels = {
            0: "anger",
            1: "fear",
            2: "joy",
            3: "love",
            4: "sadness",
            5: "surprise"
        }
        return model, tokenizer, labels

    # method for components
    def UiComponents(self):
        # creating a QLineEdit object
        sentence_input = QLineEdit("Enter you Sentence in here: (e.g. I am happy)", self)

        # setting geometry
        sentence_input.setGeometry(80, 40, 300, 40)

        # creating a label
        emotion_label = QLabel("Sentence's Emotion will be displayed in here", self)

        # setting geometry to the label
        emotion_label.setGeometry(80, 80, 300, 60)

        # setting word wrap property of label
        emotion_label.setWordWrap(True)

        img_label = QLabel(self)
        img_label.setGeometry(80, 140, 300, 300)
        pixmap = QPixmap('./image/emotion.jpg')
        pixmap = pixmap.scaled(300, 300, QtCore.Qt.KeepAspectRatio)
        img_label.setPixmap(pixmap)

        # adding action to the line edit when enter key is pressed
        sentence_input.returnPressed.connect(lambda: do_action())

        # method to do action
        def do_action():
            # getting text from the line edit
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

    # create pyqt5 app
    App = QApplication(sys.argv)

    # create the instance of our Window
    window = Window()

    # start the app
    sys.exit(App.exec())