import streamlit as st
import joblib

# to run
# streamlit email_web_app.py

st.write("basic email")

subject = st.text_input("Email subject: ")
text = st.text_input("Email text: ")

if st.button("send"):
    input = [subject + ' ' + text]

    spam_model = joblib.load('stage1.pkl')

    spam_prediction = spam_model.predict(input)

    if(spam_prediction == 1):
        class_spam = joblib.load('stage2.pkl')
        class_prediction = class_spam.predict(input)

        st.write(f"Email is spam, its type is : {class_prediction}")
    else:
        st.write("Email is legit")