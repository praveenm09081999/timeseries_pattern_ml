from flask import Flask, render_template, request, redirect, url_for
from flask_wtf import FlaskForm
from wtforms import FileField
import os
import pandas as pd
from Pattern.Univariant.univarianttest import predictUniModel
from Pattern.Multivariant.multivariant1 import predictModel1
from Pattern.Multivariant.multivariant2 import predictModel2
app = Flask(__name__)
app.config['SECRET_KEY'] = "praveen"

class FileUploadForm(FlaskForm):
    csv_file = FileField('CSV File')
# Homepage
@app.route('/', methods=['GET', 'POST'])
def home():
    form = FileUploadForm()

    if form.validate_on_submit():
        # Save the uploaded file
        file = form.csv_file.data
        filename = os.path.join('uploads', file.filename)
        file.save(filename)

        # Read the CSV file
        df = pd.read_csv(filename)

        no_column = df.shape[1]
        if no_column == 2:
            result = predictUniModel(df)
            
        else:
            result = predictModel1(df) + predictModel2(df) 
            #multivariant
        os.remove(filename)
        return render_template('result.html', result=result)

    return render_template('index.html', form=form)

if __name__ == '__main__':
    app.run(debug=True)
