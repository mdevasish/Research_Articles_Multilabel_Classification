# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 15:06:34 2021

@author: mdevasish
"""

import pandas as pd
import numpy as np
import joblib
import nltk
import re
import tensorflow as tf
from tensorflow.keras.layers import Embedding,Dropout,Bidirectional
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot,Tokenizer
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

stop_words = set(nltk.corpus.stopwords.words('english'))

def read_files():
    '''
    Function to read the train and test datset
    
    Returns:
        
        df   : Train data
        test : Test data
    
    '''
    df = pd.read_csv('./Train.csv')
    test = pd.read_csv('./Test.csv')
    return df,test

def preprocess_text(text):
    '''
    Function to clean the textual data
    
    Input Parameters :
        text  : Input text
    
    Returns :
        sentence : Cleaned text
    '''
    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', text)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

if __name__ == '__main__':
    df,test = read_files()
    target = ['Analysis of PDEs', 'Applications','Artificial Intelligence', 'Astrophysics of Galaxies','Computation and Language', 'Computer Vision and Pattern Recognition',
              'Cosmology and Nongalactic Astrophysics','Data Structures and Algorithms', 'Differential Geometry','Earth and Planetary Astrophysics','Fluid Dynamics',
              'Information Theory', 'Instrumentation and Methods for Astrophysics','Machine Learning', 'Materials Science','Methodology','Number Theory',
              'Optimization and Control', 'Representation Theory', 'Robotics','Social and Information Networks', 'Statistics Theory',
              'Strongly Correlated Electrons', 'Superconductivity','Systems and Control']
    topic_col = ['Computer Science','Mathematics', 'Physics','Statistics']
    
    
    embedding_vector_features=100
    max_len = 80
    
    X = []
    sentences = list(df["ABSTRACT"])
    for sen in sentences:
        X.append(preprocess_text(sen))

    y = df[target].values
    X_train,X_val,y_train,y_val = train_test_split(X,y,test_size = 0.2,random_state = 2021)
    
    tokenizer = Tokenizer(num_words=7000)
    tokenizer.fit_on_texts(X_train)

    X_train = tokenizer.texts_to_sequences(X_train)
    X_val = tokenizer.texts_to_sequences(X_val)
    
    voc_size = len(tokenizer.word_index) + 1
    
    X_train = pad_sequences(X_train, padding='post', maxlen=max_len)
    X_val = pad_sequences(X_val, padding='post', maxlen=max_len)
    
    
    embeddings_dictionary = dict()

    glove_file = open('./glove/glove.6B.100d.txt', encoding="utf8")

    for line in glove_file:
        records = line.split()
        word = records[0]
        vector_dimensions = np.asarray(records[1:], dtype='float32')
        embeddings_dictionary[word] = vector_dimensions
    glove_file.close()
    
    embedding_matrix = np.zeros((voc_size, 100))
    for word, index in tokenizer.word_index.items():
        embedding_vector = embeddings_dictionary.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector
            
    tf.keras.backend.clear_session()
    
    deep_inputs = Input(shape=(max_len,))
    embedding_layer = Embedding(voc_size, 100, weights=[embedding_matrix], trainable=False)(deep_inputs)
    LSTM_Layer_1 = LSTM(128)(embedding_layer)
    dense_layer_1 = Dense(25, activation='sigmoid')(LSTM_Layer_1)
    model = Model(inputs=deep_inputs, outputs=dense_layer_1)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[f1_m])
    # model=Sequential()
    # model.add(Embedding(voc_size,embedding_vector_features,input_length=max_len))
    # model.add(Bidirectional(LSTM(100)))
    # model.add(Dense(25,activation='softmax'))
    # model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=[f1_m,precision_m,recall_m])
    model.fit(X_train,y_train,batch_size = 128,epochs = 50,verbose = 1,validation_split = 0.2)