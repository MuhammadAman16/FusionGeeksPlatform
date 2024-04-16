from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        val1 = float(request.form['bedrooms'])
        val2 = float(request.form['bathrooms'])
        val3 = float(request.form['floors'])
        val4 = float(request.form['yr_built'])
        arr = np.array([val1, val2, val3, val4])
        pred = model.predict([arr])
        return jsonify({'prediction': int(pred)})
    except ValueError:
        return jsonify({'error': 'Invalid input. Please provide numeric values for all fields.'}), 400

if __name__ == '__main__':
    app.run(debug=True)
