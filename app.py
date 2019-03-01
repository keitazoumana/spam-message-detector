from flask import Flask,render_template,url_for,request
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('message.html')

@app.route('/predict', methods=['POST'])
def predict():
    df = pd.read_csv('spam.csv', encoding="latin-1")

    #Features and labels
    df['label'] = df['v1'].map({'ham': 0, 'spam': 1})
    X = df['v2']
    y = df['label']

    #Extract Feature with CountVectorizer
    cv = CountVectorizer()
    X = cv.fit_transform(X)  # Fit the Data

    #Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # The classifier
    from sklearn.naive_bayes import MultinomialNB

    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    clf.score(X_test, y_test)

    #Saved Model
    joblib.dump(clf, 'spam_detector_model.pkl')
    spam_detector_model = open('spam_detector_model.pkl', 'rb')
    clf = joblib.load(spam_detector_model)

    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)

    return render_template('prediction.html', prediction=my_prediction)

if __name__ == '__main__':
    app.run(debug=True)