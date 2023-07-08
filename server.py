from flask import Flask, json, jsonify, render_template, request
import numpy as np
import _pickle as cPickle

with open('Diabetes.model', 'rb') as f:
    rf = cPickle.load(f)
    

app = Flask(__name__)


def dataPrep(X):
    data = np.array(X)
    
    
    return 1

def guess(X):
    print(X)
    #preds = rf.predict(X)

    return 1

@app.route('/')
def hello():
    return render_template('index.html')

@app.route('/guess', methods=['POST'])
def addOne():
    newGuess = request.get_json()
    return jsonify({'guess' : guess(dataPrep(newGuess))})

if __name__ == "__main__":
    app.run(debug=True)