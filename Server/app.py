from flask import Flask, jsonify, render_template, request
import numpy as np
import _pickle as cPickle

print("Loading Model")

with open('Diabetes.model', 'rb') as f:
    rf = cPickle.load(f)
    
print("Model Loaded")

app = Flask(__name__)


def dataPrep(X):
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
        
    data = np.append(data, X["bmi"])

    data = data.astype(float)
    print(data)
    return data.reshape(1, -1)

def guess(X):
    print("Guessing")
    preds = rf.predict(X)
    print(preds[0])
    return preds[0]

@app.route('/')
def hello():
    print("Home Accessed")
    return render_template('index.html')

@app.route('/success')
def success():
    print("Success Accessed")
    return render_template('success.html')

@app.route('/guess', methods=['POST'])
def addOne():
    newGuess = request.get_json()
    print(newGuess)
    if guess(dataPrep(newGuess)) == 1:
        nGuess = "HIGH"
    else:
        nGuess = "LOW"
    return jsonify({'guess' : nGuess})

if __name__ == "__main__":
    app.run(debug=True,host="0.0.0.0" , port=5003)