from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

joblib_model = joblib.load('')