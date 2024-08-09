from flask import Flask, request, render_template
import pickle
import numpy as np
from sklearn.datasets import load_iris

app = Flask(__name__)
iris = load_iris()

with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('flower.html')

@app.route('/predict',methods=['POST'])
def  predict():
    sepal_length=float(request.form['sepal_length'])
    sepal_width=float(request.form['sepal_width'])
    petal_length=float(request.form['petal_length'])
    petal_width=float(request.form['petal_width'])

    print('@@@@@@@@@@@@@@@@@@',sepal_length)

    features=np.array([[sepal_length,sepal_width,petal_length,petal_width]])
    prediction=model.predict(features)
    species=iris.target_names[prediction[0]]

    print('@@@@@@@@@@@@@@@@@@',species)

    return render_template('flower.html',prediction_result=species)

if __name__ == '__main__':
    app.run(debug=True)
