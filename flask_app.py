from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle


# Load the trained ML model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/ReturnHome')
def return_home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the form
    input_data = [float(x) for x in request.form.values()]
    # input_data = np.ndarray(input_data).reshape(1, -1)
    input_array = np.array([input_data])

    # list of attributes
    inputs = ["Age in years", "Diabetes", "TG(Triglycerides mg/dl)", "HT(HyperTension)", "HDL(High-density lipoprotein in mg/dl)", "AC(Abdominal Circumference in cm)"]

    print(type(input_array))
    # Make a prediction using the loaded model
    probs = model.predict_proba(input_array)[0, 1]  # get the probability of positive class
    label = pd.cut([probs * 100], bins=[0, 50, 75, 100], labels=['Low', 'Medium', 'High'])
    print("Predicted label:", label[0])
    prediction = label[0]

    # Render the results template with the prediction
    return render_template('results.html', prediction=prediction, input_data=input_data, inputs=inputs)


if __name__ == '__main__':
    app.run(debug=True)


