# flask, scikit-learn, pandas, pickle-mixin, flask_cors

import pandas as pd
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
data = pd.read_csv("Cleaned_data.csv")
pipe = pickle.load(open("RidgeModel.pkl",'rb'))

@app.route('/')
def index():
    locations = sorted(data['location'].unique())
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['post'])
def predict():
    location = request.form.get('location')
    bhk= request.form.get('bhk')
    bathroom = request.form.get('bathroom')
    sqft = request.form.get('sqft')
    print(location, bhk, bathroom, sqft)
    input = pd.DataFrame([[location, sqft, bathroom, bhk]], columns=['location','sqft','bathroom','bhk'])
    prediction = pipe.predict(input)[0] * 1e5

    return str(np.round(prediction, 2))

if __name__ == "__main__":
    app.run(debug=True, port=5001)