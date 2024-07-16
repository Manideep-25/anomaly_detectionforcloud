from flask import Flask, render_template, request, redirect, url_for
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Load the training data
        df = pd.read_csv("./anomaly_data.csv")

        # Data preprocessing and training
        df.drop_duplicates(inplace=True)
        sample_fraction = 0.1
        df_sampled, _ = train_test_split(df, test_size=(1 - sample_fraction), stratify=df['Label'], random_state=42)
        train, test = train_test_split(df_sampled, test_size=0.4, stratify=df_sampled['Label'], random_state=42)
        train = train[~train.duplicated()]
        test = test[~test.duplicated()]

        def scale_dataset(dataframe):
            X = dataframe.iloc[:, :-2].values
            y = dataframe.iloc[:, -1].values
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            return X, y

        X_train, y_train = scale_dataset(train)
        svm_model = SVC()
        svm_model.fit(X_train, y_train)

        # Load the uploaded test data
        uploaded_df = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        X_test, y_test = scale_dataset(uploaded_df)

        # Make predictions on the uploaded test set
        y_pred = svm_model.predict(X_test)

        # Check for threats or malware
        if np.any(y_pred == 1):
            result = "There is a threat or malware"
            result_class = "threat"
        else:
            result = "There is no threat or malware"
            result_class = "no-threat"

        return render_template('index.html', result=result, result_class=result_class)

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
