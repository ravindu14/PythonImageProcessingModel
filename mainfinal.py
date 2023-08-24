from flask import Flask, render_template, request, jsonify
app = Flask(__name__)

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np

from keras import backend as K
import keras.models as models
import keras.optimizers as optimizers
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Convolution2D, ZeroPadding2D, AveragePooling2D
from keras.layers.advanced_activations import PReLU
import base64
import os
import cv2
    
    
def load_model():
    
    print('hit load model')
    model = models.Sequential()
    
    model.add(Conv2D(256, (3, 3), input_shape=(50,50,1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten()) 
    
    model.add(Dense(64))
    
    model.add(Dense(8))
    model.add(Activation('sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    model.load_weights('Deficiency_V1.h5')
    
    print('last model')
    return model

@app.route('/')
def student():
   return render_template('student.html')
   
   K.clear_session()


        
@app.route('/deficiency',methods = ['POST', 'GET'])
def food():
    
    print('came here')
    model = load_model()
    
    print('after model')
    target_dict={0:'Citrus_Nitrogen_Deficiency', 1: 'Citrus_Phosphorous_Deficiency', 2: 'Citrus_Pottassium_Deficiency', 3: 'Groundnut_Nitrogen_Deficiency', 4: 'Groundnut_Pottassium_Deficiency', 5: 'Groundnut_Sulfur_Deficiency', 6: 'Guava_Nitrogen_Deficiency', 7: 'Guava_Pottassium_Deficiency'}
	
    User_json = request.json
    
    encrypted_string =  User_json['base_string']
    
    
    imgdata = base64.b64decode(encrypted_string)
    filename = 'test_image.jpg' 
    with open(filename, 'wb') as f:
        f.write(imgdata)
    
    img = cv2.imread('test_image.jpg')
    test_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (height, width) = img.shape[:2]
    
    #imgdata[0:50,0:width] = [0,0,255]
    #imgdata[50:80,0:100] = [0,255,255]
    
    
    test_img=cv2.resize(test_img,(50,50))
    test_img=test_img/255.0
    test_img=test_img.reshape(-1,50,50,1)
    result=model.predict([test_img])
        
    K.clear_session()
    label=target_dict[np.argmax(result)]
        
    
        
    results = [
    {
        "predictedResult": label
    }
    ]
    return jsonify(results=results)    

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5000)
