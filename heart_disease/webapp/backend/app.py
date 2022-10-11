from flask import Flask, request
import pickle
import pandas as pd
import json

app = Flask(__name__)
MODEL = pickle.load(open('model.pkl', 'rb'))
DF = pd.read_csv('./data/clean_df.csv')

@app.route('/api/medians_by_age/<int:age>')
# Finding Median Values Given age
def medians_by_age(age):
    # Get Unique Values
    ages = DF['age'].unique()
    ages.sort()

    closest_age = ages[0]
    # Perform binary search to find closest value
    l = 0
    r = len(ages) - 1
    while l <= r:
        m = (r + l)//2
        print(m, ages[m])
        if ages[m] == age:
            closest_age = ages[m]
            break
        
        if ages[m] > age:
            r = m - 1
        elif ages[m] < age:
            l = m + 1

    if l == len(ages):
        closest_age = ages[-1]
    elif closest_age != age:
        closest_age = ages[l]
    
    # Filter Dataframe
    df_by_age = DF.loc[DF['age'] == closest_age]
    return df_by_age.median().to_json()

@app.route("/api/predict", methods=['POST'])
def predict():
    raw_data = request.get_json(force=True)
    data = {}
    for key in raw_data:
        data[key] = float(raw_data[key])

    prediction = MODEL.predict([[data['male'], data['age'], data['currentSmoker'], data['cigsPerDay'], data['BPMeds'], data['prevalentStroke'], data['prevalentHyp'], data['diabetes'], data['totChol'], data['sysBP'], data['diaBP'], data['BMI'], data['heartRate'], data['glucose']]])
    result = prediction[0]
    return {"prediction": result}

if __name__ == "__main__":
    app.run()
