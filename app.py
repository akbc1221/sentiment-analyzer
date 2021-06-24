from flask import Flask, render_template, flash, request, url_for, redirect, session
import numpy as np
import pandas as pd
import re
import os
import tensorflow as tf
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import load_model

IMAGE_FOLDER = os.path.join('static', 'img')

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = IMAGE_FOLDER


def init():
    global model, graph
    model = load_model('./model_IMDB.h5')
    graph = tf.get_default_graph()

# Code for Sentiment Analysis


@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template("home.html")


@app.route('/predict_sentiment', methods=['POST', "GET"])
def sent_anly_prediction():
    if request.method == 'POST':
        text = request.form['text']
        Sentiment = ''
        max_review_length = 500
        word_to_id = imdb.get_word_index()
        strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
        text = text.lower().replace("<br />", " ")
        text = re.sub(strip_special_chars, "", text.lower())

        words = text.split()
        x_test = [[word_to_id[word] if (
            word in word_to_id and word_to_id[word] <= 20000) else 0 for word in words]]
        # Should be same which you used for training data
        x_test = sequence.pad_sequences(x_test, maxlen=500)
        vector = np.array([x_test.flatten()])
        with graph.as_default():
            probability = model.predict(np.array([vector][0]))[0][0]
            class1 = model.predict_classes(np.array([vector][0]))[0][0]
        if class1 == 0:
            sentiment = 'Negative'
            img_filename = os.path.join(
                app.config['UPLOAD_FOLDER'], 'neg_emoji.png')
        else:
            sentiment = 'Positive'
            img_filename = os.path.join(
                app.config['UPLOAD_FOLDER'], 'pos_emoji.png')

    return render_template('home.html', text=text, sentiment=sentiment, probability=probability, image=img_filename)


if __name__ == "__main__":
    init()
    app.run()
