from flask import Flask, render_template, request
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Simple machine learning model
model = LogisticRegression()


# Homepage
@app.route('/')
def home():
    return render_template('index.html')


# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input data from the form
        feature1 = float(request.form['feature1'])
        feature2 = float(request.form['feature2'])

        # Make prediction using the model
        prediction = model.predict([[feature1, feature2]])

        # Display the result
        result = 'Class 1' if prediction[0] == 1 else 'Class 0'
        return render_template('result.html', result=result)


if __name__ == '__main__':
    app.run(debug=True)
