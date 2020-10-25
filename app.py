from flask import Flask, render_template, request
import os
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,OneHotEncoder
import tensorflow as tf

app = Flask(__name__)

h5_path = os.path.join("models", "trained_por_G1.h5")
model = tf.keras.models.load_model(h5_path)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["get", "post"])
def predict():
    inputs = []
    print("0")
    inputs.append(request.form["school"])
    print("1")
    inputs.append(request.form["sex"])
    inputs.append(request.form["school"])
    inputs.append(request.form["age"])
    inputs.append(request.form["address"])
    inputs.append(request.form["famsize"])
    inputs.append(request.form["Pstatus"])
    inputs.append(request.form["Medu"])
    inputs.append(request.form["Fedu"])
    inputs.append(request.form["Mjob"])
    inputs.append(request.form["Fjob"])
    inputs.append(request.form["reason"])
    inputs.append(request.form["guardian"])
    inputs.append(request.form["traveltime"])
    inputs.append(request.form["studytime"])
    inputs.append(request.form["failures"])
    inputs.append(request.form["schoolsup"])
    inputs.append(request.form["famsup"])
    inputs.append(request.form["paid"])
    inputs.append(request.form["activities"])
    inputs.append(request.form["nursery"])
    inputs.append(request.form["higher"])
    inputs.append(request.form["internet"])
    inputs.append(request.form["romantic"])
    inputs.append(request.form["famrel"])
    inputs.append(request.form["goout"])
    inputs.append(request.form["Dalc"])
    inputs.append(request.form["Walc"])
    inputs.append(request.form["health"])
    try:
        inputs.append(request.form["absences"])
    except:
        inputs.append(0)
    print("2")

    por_path = os.path.join("Student_alcohol_consumption", "student-por.csv")
    data = pd.read_csv(por_path)

    data.append(inputs)

    print("3")

    # Generate our categorical variable list
    encode_cat = data.dtypes[data.dtypes == "object"].index.tolist()

    # Check the number of unique values in each column
    data[encode_cat].nunique()

    # Create the OneHotEncoder instance
    enc = OneHotEncoder(sparse=False)

    # Fit the encoder and produce encoded DataFrame
    encode_df = pd.DataFrame(enc.fit_transform(data[encode_cat]))

    # Rename encoded columns
    encode_df.columns = enc.get_feature_names(encode_cat)

    # Merge the two DataFrames together and drop the Country column
    data = data.merge(encode_df,left_index=True,right_index=True).drop(encode_cat, 1)
    data = data.drop(["G1", "G2", "G3"],1).values
    data = pd.DataFrame(data)
    print(data)

    # Create a StandardScaler instance
    scaler = StandardScaler()

    # Fit the StandardScaler
    data_scaler = scaler.fit(data)
    data_scaled = data_scaler.transform(data)
    print(data_scaled)
    data_scaled = pd.DataFrame(data_scaled)
    scaled_inputs = data_scaled.tail(1)
    scaled_inputs = np.array(scaled_inputs)
    print(scaled_inputs)

    result="Pass"

    answer = model.predict(scaled_inputs.reshape(1,56))
    print(answer)
    
    return render_template("index.html", result=str(answer))

if __name__ == '__main__':
    app.run(debug=True)