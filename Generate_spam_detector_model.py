import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.externals import joblib

df = pd.read_csv('spam.csv', encoding="latin-1")
df['label'] = df['v1'].map({'ham': 0, 'spam': 1})

X = df['v2']
y = df['label']

cv = CountVectorizer()
X = cv.fit_transform(X) # Fit the Data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#Naive Bayes Classifier
clf = MultinomialNB()
clf.fit(X_train,y_train)
clf.score(X_test,y_test)
y_pred = clf.predict(X_test)
#print(classification_report(y_test, y_pred))

# Persist our model
joblib.dump(clf, 'spam_detector_model.pkl')