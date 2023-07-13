from flask import Flask, jsonify, render_template, request
import numpy as np
import _pickle as cPickle
import tensorflow as tf

print("Loading Simple Model")

with open('simple.model', 'rb') as f:
    rfS = cPickle.load(f)
    
print("Loading Complex Model")
rfC = tf.keras.models.load_model('complex.h5')

    
print("Model Loaded")

app = Flask(__name__)

count = 0

def dataPrepComplex(X):
    data = np.empty(shape=(0, 1))
    
    data = np.append(data, X["age"])
    
    try:
        if X["hypertension"] == "on":
            data = np.append(data, 1)
    except:
        data = np.append(data, 0)
        
    try:
        if X["heart_disease"] == "on":
            data = np.append(data, 1)
    except:
        data = np.append(data, 0)

    data = np.append(data, X["bmi"])
    data = np.append(data, X["HbA1c_level"])
    data = np.append(data, X["blood_glucose_level"])
    
    if X["gender"] == "Female":
        print("Female")
        data = np.append(data, 1)
        data = np.append(data, 0)
    else:
        print("Male")
        data = np.append(data, 0)
        data = np.append(data, 1)

    if X["smoking_history"] == "Never":
        print("never")
        data = np.append(data, 0)
        data = np.append(data, 1)
        data = np.append(data, 0)
    elif X["smoking_history"] == "No Info":
        print("No Info")
        data = np.append(data, 0)
        data = np.append(data, 1)
        data = np.append(data, 0)
    elif X["smoking_history"] == "current":
        print("current")
        data = np.append(data, 1)
        data = np.append(data, 0)
        data = np.append(data, 0)
    elif X["smoking_history"] == "former":
        print("former")
        data = np.append(data, 0)
        data = np.append(data, 0)
        data = np.append(data, 1)
        
    


    data = data.astype(float)
    print(data)
    return data.reshape(1, -1)

def dataPrepSimple(X):
    data = np.empty(shape=(0, 1))
    
    data = np.append(data, X["age"])
    
    try:
        if X["hypertension"] == "on":
            data = np.append(data, 1)
    except:
        data = np.append(data, 0)
        
    try:
        if X["heart_disease"] == "on":
            data = np.append(data, 1)
    except:
        data = np.append(data, 0)

    if X["gender"] == "Female":
        data = np.append(data, 1)
        data = np.append(data, 0)
    else:
        print("Male")
        data = np.append(data, 0)
        data = np.append(data, 1)

    if X["smoking_history"] == "Never":
        print("never")
        data = np.append(data, 0)
        data = np.append(data, 1)
        data = np.append(data, 0)
    elif X["smoking_history"] == "No Info":
        print("No Info")
        data = np.append(data, 0)
        data = np.append(data, 1)
        data = np.append(data, 0)
    elif X["smoking_history"] == "current":
        print("current")
        data = np.append(data, 1)
        data = np.append(data, 0)
        data = np.append(data, 0)
    elif X["smoking_history"] == "former":
        print("former")
        data = np.append(data, 0)
        data = np.append(data, 0)
        data = np.append(data, 1)
        
    data = np.append(data, X["bmi"])

    data = data.astype(float)
    print(data)
    return data.reshape(1, -1)

def simplePredict(X):
    print("Guessing")
    preds = rfS.predict(X)
    return preds[0]

def complexPredict(X):
    print("Guessing")
    print(X)
    preds = rfC.predict(X)
    return preds[0]

@app.route('/')
def hello():
    count =+ 1
    print(count)
    return render_template('index.html')

@app.route('/success')
def success():
    return render_template('success.html')

@app.route('/simple')
def simple():
    return render_template('simple.html')

@app.route('/complex')
def complex():
    return render_template('complex.html')


@app.route('/simple/guess', methods=['POST'])
def simpleGuess():
    newGuess = request.get_json()
    if simplePredict(dataPrepSimple(newGuess)) == 1:
        nGuess = "HIGH"
    else:
        nGuess = "LOW"
    return jsonify({'guess' : nGuess})

@app.route('/complex/guess', methods=['POST'])
def complexGuess():
    newGuess = request.get_json()
    prediction = complexPredict(dataPrepComplex(newGuess))
    print(prediction)
    if prediction == 1:
        nGuess = "HIGH"
    else:
        nGuess = "LOW"
    return jsonify({'guess' : nGuess})

if __name__ == "__main__":
    app.run(debug=True,host="0.0.0.0" , port=5003)