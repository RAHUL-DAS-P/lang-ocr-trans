import sys
import requests
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox, QFileDialog
from PyQt5.QtGui import QPixmap
import speech_recognition as sr
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pytesseract
from PIL import Image
from googletrans import Translator
from language_tool_python import LanguageTool


class LanguageApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Language App")
        self.setGeometry(100, 100, 800, 600)  # Adjusted window size

        layout = QVBoxLayout()
        # Define button styles
        button_style = """
        QPushButton {
        background-color: #4CAF50;
        border: none;
        color: white;
        padding: 15px 24px;  /* Increase padding-top and padding-bottom to increase height */
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 25px;
        margin: 4px 2px;
        cursor: pointer;
        min-height: 30px;
        border-radius: 15px;  /* Increase border-radius for more rounded corners */
        box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);  /* Add shadow */
        }

        QPushButton:hover {
            background-color: #45a049;
        }
        """

        self.text_label = QLabel("Enter Text:")
        # Set maximum height for the label
        self.text_label.setMaximumHeight(100)
        layout.addWidget(self.text_label)

        self.text_entry = QLineEdit()
        # Set minimum size for the text input box
        self.text_entry.setMinimumSize(400, 200)
        layout.addWidget(self.text_entry)

        self.clear_button = QPushButton("Clear Text")
        self.clear_button.clicked.connect(self.clear_text)
        self.clear_button.setStyleSheet(button_style)
        layout.addWidget(self.clear_button)

        self.audio_button = QPushButton("Input Audio")
        self.audio_button.clicked.connect(self.input_audio)
        self.clear_button.setStyleSheet(button_style)
        layout.addWidget(self.audio_button)

        self.image_button = QPushButton("Input Image")
        self.image_button.clicked.connect(self.input_image)
        self.clear_button.setStyleSheet(button_style)
        layout.addWidget(self.image_button)

        self.detect_button = QPushButton("Detect Language")
        self.detect_button.clicked.connect(self.detect_language)
        self.clear_button.setStyleSheet(button_style)
        layout.addWidget(self.detect_button)
        self.grammar_button = QPushButton("Check Grammar")
        self.grammar_button.clicked.connect(self.check_grammar)
        self.clear_button.setStyleSheet(button_style)
        layout.addWidget(self.grammar_button)

        self.hate_button = QPushButton("Detect and Remove Hate Speech")
        self.hate_button.clicked.connect(self.detect_and_remove_hate_speech)
        self.clear_button.setStyleSheet(button_style)
        layout.addWidget(self.hate_button)
        self.lang_input = QLineEdit()
        self.lang_input.setPlaceholderText("Enter language code or name")
        layout.addWidget(self.lang_input)

        self.translate_button = QPushButton("Translate")
        self.translate_button.clicked.connect(self.translate_text)
        self.clear_button.setStyleSheet(button_style)
        layout.addWidget(self.translate_button)

        self.result_label = QLabel()
        layout.addWidget(self.result_label)

        self.setLayout(layout)

    def detect_language(self):
        user_input = self.text_entry.text()
        if user_input:
            detected_language = predict_language(user_input)
            self.show_popup(f"The detected language is: {detected_language}")
        else:
            self.show_error_popup("Please enter some text.")

    def translate_text(self):
        user_input = self.text_entry.text()
        try:
            if user_input:
                dest_lang = self.lang_input.text().lower()  # Convert to lowercase
                if dest_lang:
                    translated_text = translate_text(user_input, dest_lang)
                    self.show_popup(f"Translated text: {translated_text}")
                else:
                    self.show_error_popup(
                        "Please enter a destination language.")
            else:
                self.show_error_popup("Please enter some text.")
        except Exception as e:
            self.show_error_popup(f"An error occurred: {str(e)}")

    def check_grammar(self):
        user_input = self.text_entry.text()
        if user_input:
            corrected_text = correct_grammar(user_input)
            self.show_popup(f"Corrected text: {corrected_text}")
        else:
            self.show_error_popup("Please enter some text.")

    def input_audio(self):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("Listening...")
            audio_data = recognizer.listen(source)
            try:
                dest_lang = self.lang_input.text() + "-IN"
                if dest_lang:
                    text = recognizer.recognize_google(
                        audio_data, language=dest_lang)
                    self.text_entry.setText(text)
                else:
                    text = recognizer.recognize_google(audio_data)
                    self.text_entry.setText(text)
            except sr.UnknownValueError:
                self.show_error_popup("Could not understand audio")
            except sr.RequestError as e:
                self.show_error_popup(
                    f"Could not request results from Google Speech Recognition service; {e}")

    def input_image(self):
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("Images (*.png *.jpg *.bmp)")
        # Use QFileDialog.Detail instead of QFileDialog.Icon
        file_dialog.setViewMode(QFileDialog.Detail)
        file_dialog.setDirectory('.')
        file_dialog.fileSelected.connect(self.on_file_selected)
        file_dialog.exec_()

    def detect_and_remove_hate_speech(self):
        user_input = self.text_entry.text()
        if user_input:
            self.show_popup("Hate speech detection coming soon")
        else:
            self.show_error_popup("Please enter some text.")

    def on_file_selected(self, file):
        text = extract_text_from_image(file)
        if text:
            self.text_entry.setText(text)
        else:
            self.show_error_popup("Unable to extract text from image.")

    def clear_text(self):
        self.text_entry.clear()

    def show_popup(self, message):
        QMessageBox.information(self, "Success", message, QMessageBox.Ok)

    def show_error_popup(self, message):
        QMessageBox.warning(self, "Error", message, QMessageBox.Ok)


# Load the data
data = pd.read_csv(r'dataset.csv')

# Separate features and target variable
x = np.array(data["Text"])
y = np.array(data["language"])

# Vectorize the text data
cv = CountVectorizer()
X = cv.fit_transform(x)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

# Train the Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Function to predict language from user input


def predict_language(user_input):
    user_data = cv.transform([user_input]).toarray()
    output = model.predict(user_data)
    return output[0]

# Function to translate text


def translate_text(text, dest_lang):
    translator = Translator()
    translation = translator.translate(text, dest=dest_lang)
    return translation.text

# Function to correct grammar


def correct_grammar(text):
    tool = LanguageTool('en-US')
    matches = tool.check(text)
    return tool.correct(text)

# Function to extract text from image


def extract_text_from_image(image_path):
    try:
        text = pytesseract.image_to_string(
            Image.open(image_path), lang="hin+eng+fra+deu+rus+ara+spa+por+ita+hin+ben+jpn+kor")
        print(text)
        return text
    except Exception as e:
        print(e)
        return None


if __name__ == '__main__':
    app = QApplication(sys.argv)
    lang_app = LanguageApp()
    lang_app.show()
    sys.exit(app.exec_())
