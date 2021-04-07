from flask import Flask, request, render_template
import pickle
import numpy as np


app = Flask(__name__)
model = pickle.load(open('salary_prediction.sav', 'rb'))

@app.route('/')
def hello():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    
    int_features = [float(x) for x in request.form.values()]
    feature = np.array(int_features)
    prediction = model.predict(feature.reshape(-1,1))

    output = round(prediction[0][0], 2)
    year = round(feature[0],2)
    return render_template('index.html', 
                           prediction_text=f'Employee Salary for {year} years of experience should be $ {output}')
 
if __name__ == '__main__':
    app.run()
