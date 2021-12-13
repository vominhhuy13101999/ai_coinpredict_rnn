import os
import sys
import rnn
import pandas as pd
# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Some utilites
import numpy as np
# from util import base64_to_pil



# Declare a flask app
app = Flask(__name__,template_folder='html')
MODEL_PATH = 'models/eth_usd.h5'

def load_model(name):
    model = tf.keras.models.load_model('model/weekly_10_{}.h5'.format(name))
    return model
def model_predict(name ):
    dataset = pd.read_excel('file/{}_weekly_return.xlsx'.format(name))
    model=load_model(name)
    s=dataset.drop([0])
    s=s.reset_index().drop(columns=["index",'Unnamed: 0','Unnamed: 0.1'])
    open_process=rnn.return_processing()
    close_process=rnn.return_processing()
    open_process.fit(dataset.Open)
    close_process.fit(dataset.Close)
    train_size = int(len(s) * 0.9)
    look_back = 10

    trainX, trainY = rnn.create_dataset(s, look_back,train_size)
    testX, testY = rnn.create_dataset(s, look_back,train_size,test=True)
    trainX = np.reshape(trainX, (trainX.shape[0], look_back+1,1))
    testX = np.reshape(testX, (testX.shape[0], look_back+1,1))  
    testPredict = model.predict(testX)

    testPredict = testPredict
    testY_ = [testY]
    test_=close_process.inverse(testPredict.reshape(-1,),head=False,start=0)
    model=load_model(name)

    low=np.zeros((1,))
    low=np.concatenate((low, testY), axis=0)

    high=np.zeros((1,))
    high=np.concatenate((high, testPredict.reshape(-1,)), axis=0)
    high_norm=np.zeros((1,))
    high_norm=np.concatenate((high_norm, test_.reshape(-1,)), axis=0)
    dic={"Y":low,"predict":high,"predict_normalize":high_norm}

    d=pd.DataFrame.from_dict(dic)
    return dic

@app.route('/', methods=['GET'])
def home():
    # Main page
    return render_template('homepage.html')



@app.route('/bitcoin', methods=['GET'])
def bitcoin():
    print(model_predict("btc" ))
    return render_template('bitcoin.html')

@app.route('/eth', methods=['GET'])
def eth():
    print(model_predict("eth" ))
    return render_template('etherum.html')

@app.route('/ada', methods=['GET'])
def ada():
    return render_template('cardano.html')

@app.route('/doge', methods=['GET'])
def doge():
    return render_template('dogecoin.html')

if __name__ == '__main__':
    print('Model loaded. Check http://127.0.0.1:5000/')

    # app.run(port=5002, threaded=False)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()