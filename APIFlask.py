import pickle
import numpy as np
import json
from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "Hello World"

with open("model.pickle", "rb") as f:
    model_dict = pickle.load(f)

model = model_dict["model"]

def model_pred(data_coordinates_array):
    pred = model.predict([np.asarray(data_coordinates_array)])
    return pred


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    data_coordinates_array = json.loads(data)
    prediction = model_pred(data_coordinates_array["dc"])
    # print(data["dc"])
    return jsonify({"Prediction" : prediction[0]})

# @app.post("/predict")
# def predict(itemJSON:ImageJSON):
#     data_coordinates_array = itemJSON.model_dump()["dc"]
#     prediction = model_pred(data_coordinates_array)
#     # return FileResponse(path)
#     return {"Prediction" : prediction[0]}

app.run(host="0.0.0.0", port="8000", debug=True)