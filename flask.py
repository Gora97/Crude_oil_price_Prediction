from flask import Flask, render_template, request
import pandas as pd
from joblib import load
import plotly.express as px
from prophet import Prophet

app = Flask(__name__)

# Load the saved Prophet model using joblib
model_filename = 'path_to_your_model_file.joblib'
model = load(model_filename)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        selected_date = pd.to_datetime(request.form['selected_date'])
        future = pd.DataFrame({'ds': [selected_date]})
        forecast = model.predict(future)
        predicted_price = forecast.loc[0, 'yhat']

        return render_template('index.html', predicted_price=predicted_price)

    return render_template('index.html', predicted_price=None)

if __name__ == '__main__':
    app.run(debug=True)
