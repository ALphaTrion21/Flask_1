import os
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create the flask app
app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), 'templates'))

# Load the pickle module
model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "model.plr")
with open(model_path, "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    return render_template("index.html", prediction_text="The Flower Species is {}".format(prediction))

if __name__ == '__main__':
    app.run(debug=True)
