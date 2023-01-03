from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('model.pkl', 'rb'))
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict' , methods = ['POST'])
def predict_churn():
    Income = request.form.get('Income')
    Grade = request.form.get('Grade')
    ReportingDayCount = request.form.get('ReportingDayCount')
    Avg_Business_Value = request.form.get('Avg Business Value')
    Gender_1 = request.form.get('Gender_1.0')
    Education_Level_1 = request.form.get('Education_Level_1')
    Education_Level_2 = request.form.get('Education_Level_2')
    QuarterlyRating_2 = request.form.get('Quarterly Rating_2')
    QuarterlyRating_3 = request.form.get('Quarterly Rating_3')
    QuarterlyRating_4 = request.form.get('Quarterly Rating_4')
    driver_rating_increased_1 = request.form.get('driver_rating_increased_1')
    income_increased_1 = request.form.get('income_increased_1')
    
    #prediction
    result = model.predict(np.array([Income,Grade,ReportingDayCount,Avg_Business_Value,Gender_1,Education_Level_1,
    Education_Level_2,QuarterlyRating_2,QuarterlyRating_3,QuarterlyRating_4,driver_rating_increased_1,
    income_increased_1]).reshape((1,12)))

    if result[0] == 1:
        result = 'Driver may Churn'
    else:
        result = 'Driver may not Churn'
    return render_template('index.html', result = result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port= 8080)