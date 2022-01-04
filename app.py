import os
import sys
import rnn
import pandas as pd
from datetime import datetime
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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
btc=["%s" for i in range(33)]
eth=["%s" for i in range(33)]

# Some utilites
import numpy as np
import codecs
# from util import base64_to_pil

def modify(l,n):
    if n==0:
        file = codecs.open("html/bitcoin.html", 'r', "utf-8")
        f = open('html/bitcoin1.html', 'w')
        predict_=l[1]
    elif n==1:
        file = codecs.open("html/ethereum.html", 'r', "utf-8")
        f = open('html/ethereum1.html', 'w')
        predict_=l[1]
    else:
        file = codecs.open("html/dogecoin.html", 'r', "utf-8")
        f = open('html/dogecoin1.html', 'w')
        predict_=-l[1]
    date=l[0]
    
    real=l[2]
    real_=l[3]

    
    filer=file.read()
    filer=filer.replace("%s", str(date[-1]),1)
    if predict_[-1]>0:

        filer=filer.replace("%s", "green",1)
    else:
        filer=filer.replace("%s", "red",1)

    filer=filer.replace("%s", str(real[-1]),1)
    print(predict_)
    print(real)



    for i in range(10):
        filer=filer.replace("%s", str(date[i]),1)
        
        if real_[i]>0:
            filer=filer.replace("%s", "Increase",1)
        else:
            filer=filer.replace("%s", "Decrease",1)
        if predict_[i]>0:
            filer=filer.replace("%s", "Increase",1)
        else:
            filer=filer.replace("%s", "Decrease",1)

    f.write(filer)
    f.close()

# Declare a flask app
app = Flask(__name__,template_folder='html')

def load_model(name):
    model = tf.keras.models.load_model('model/weekly_10_{}.h5'.format(name))
    return model
def model_predict(name ):
    dataset = pd.read_excel('file/{}_weekly_return.xlsx'.format(name))
    model=load_model(name)
    
    s=dataset.drop([0])
    s=s.reset_index().drop(columns=["index",'Unnamed: 0','Unnamed: 0.1'])
    date=[datetime.strptime(str(s.Date[i]), '%Y-%m-%d %H:%M:%S').strftime('%Y.%m.%d') for i in range(len(s.Date))]
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
    test_=close_process.inverse(testPredict.reshape(-1,),head=False,start=0)

    low=np.zeros((1,))
    low=np.concatenate((low, testY), axis=0)

    high=np.zeros((1,))
    high=np.concatenate((high, testPredict.reshape(-1,)), axis=0)
    high_norm=np.zeros((1,))
    high_norm=np.concatenate((high_norm, test_.reshape(-1,)), axis=0)
    # dic={"Date":date,"Y":low,"predict":high,"predict_normalize":high_norm}

    # d=pd.DataFrame.from_dict(dic)
    return [date[-11:],high[-11:],s.Open.tolist()[-11:],low[-11:]]

@app.route('/', methods=['GET'])
def home():
    # Main page
    return render_template('homepage.html')

@app.route('/css', methods=['GET'])
def css():
    # Main page
    return render_template('bitcoin.css')

@app.route('/bitcoin', methods=['GET'])
def bitcoin():
    # print(model_predict("btc" ))
    l=model_predict("btc" )
    # print(l)
    modify(l,0)

    return render_template('bitcoin1.html')

@app.route('/eth', methods=['GET'])
def eth():
    l=model_predict("eth" )
    modify(l,1)
    return render_template('ethereum1.html')

@app.route('/ada', methods=['GET'])
def ada():
    return render_template('cardano.html')

@app.route('/doge', methods=['GET'])
def doge():
    l=model_predict("doge" )
    modify(l,2)
    return render_template('dogecoin1.html')

if __name__ == '__main__':
    print('Model loaded. Check http://157.245.1.176:8080/')

    app.run(host='0.0.0.0',port=8080, threaded=False)
    # app.run(debug=True)
    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 8080), app)
    http_server.serve_forever()