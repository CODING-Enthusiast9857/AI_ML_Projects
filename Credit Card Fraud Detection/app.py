import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st 

data = pd.read_csv('creditcard.csv')

# Separate legitimate and fraudulent transactions
data.drop_duplicates(inplace=True)

legit = data[data['Class'] == 0]
fraud = data[data.Class == 1]

legit_sample = legit.sample(n=len(fraud), random_state = 2)
data = pd.concat([legit_sample, fraud], axis=0)

X = data.drop(columns='Class', axis=1)
y = data['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

model = LogisticRegression()
model.fit(X_train, y_train)

train_acc = accuracy_score(model.predict(X_train), y_train)
test_acc = accuracy_score(model.predict(X_test), y_test)

# print("Training Score - " + str(train_acc))
# print("Testing Score - " + str(test_acc)) 

# Web Application
st.title('Credit Card Fraud Detection Model')
input_df = st.text_input("Enter all feature values")
input_df_splited = input_df.split(',')

submit = st.button('Submit')

if submit:
    features = np.asarray(input_df_splited, dtype=np.float64)
    prediction = model.predict(features.reshape(1, -1))

    if prediction[0] == 0:
        st.write("Legitimate Transaction")
    else:
        st.write("Fraudulent Transaction")                       
