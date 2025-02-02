from flask import Flask, jsonify, request, Response, render_template
import json
from waitress import serve
import pickle
import pandas as pd
import numpy as np
import random
from sklearn import decomposition
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import Normalizer
import warnings
warnings.filterwarnings("ignore")
""" 
Models trained:
    1. Decision Tree Classifier => 3.2%
    2. Decision Tree Classifier Entropy => 3.1%
    3. SVM => 5.75 (Best)
    4. xgboost => 4.41%
    5. KNN => 3.2%
    6. Random Forest Classifier => 2.9%
"""

# Preprocessing data

def preprocess(data):
    print("Original data:", data)
    # Converting categorical data to numerical data
    label_encoder = LabelEncoder()
    for i in range(14, 38):
        data[:, i] = label_encoder.fit_transform(data[:, i])
    print("Data after label encoding:", data)
    
    data1 = data[:, :14]
    normalized_data = Normalizer().fit_transform(data1)
    print("Normalized data:", normalized_data)
    
    data2 = data[:, 14:]
    df1 = np.append(normalized_data, data2, axis=1)
    print("Final preprocessed data:", df1)
    return df1


def model_predict(model, data):
    print(f"Making prediction with model: {model}")
    prediction = model.predict(data)
    print(f"Prediction result: {prediction}")
    return prediction


models = ['dtc', 'ent_dtc', 'knn', 'rfc', 'svm']
names = ['Decision Tree Classifier', 'Decition Tree Classifier Entropy',
         'KNN', 'Random Forest Classifier', 'SVM']
loaded_models = []
for model in models:
    print(f"Loading model: {model}.pkl")
    model_instance = pickle.load(open(f'../models/{model}.pkl', 'rb'))
    loaded_models.append(model_instance)
    print(f"Model {model} loaded successfully")


app = Flask(__name__, template_folder="../templates", static_folder="../static")


@app.route('/', methods=["GET"])
def index():
    print("Rendering index.html")
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    print("Received prediction request")
    data = request.get_json(force=True)['data']
    print("Raw input data:", data)
    
    data = np.array(data).reshape((1, 38))
    print("Reshaped data:", data)
    
    data = preprocess(data)
    
    predictions = []
    for i, model in enumerate(loaded_models):
        print(f"Predicting with {names[i]}")
        prediction = model_predict(model, data)[0]
        predictions.append(prediction)
        print(f"Prediction from {names[i]}: {prediction}")
    
    print("All predictions:", predictions)
    return Response(json.dumps(predictions),  mimetype='application/json')


if __name__ == '__main__':
    print("Starting the Flask server with Waitress")
    serve(app)
