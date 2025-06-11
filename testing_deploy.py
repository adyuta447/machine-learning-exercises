from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

joblib_model = joblib.load('gbr_model.joblib')

@app.route('/predict', methods=['POST'])

def predict():
    data = request.json['data'] # Ngambil data dari request JSON
    prediction = joblib_model.predict(data)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__name__':
    app.run(debug=True)