import streamlit as st
import pickle
from nltk.corpus import stopwords
import re

stop_words = set(stopwords.words('english'))
pattern = re.compile(r'\b(?:' + '|'.join(re.escape(word) for word in stop_words) + r')\b\s*', re.IGNORECASE)

def Preprocessing(text):
    text = text.lower()
    text = re.sub(r'<[^>]*>', '', text)         # remove HTML
    text = re.sub(r'http\S+|www.\S+', '', text) # remove URLs
    text = re.sub(pattern, '', text)            # remove stopwords
    text = re.sub(r'\s+', ' ', text).strip()    # collapse spaces
    return text

def preprocess_list(X):
    return [Preprocessing(x) for x in X]

pipe = pickle.load(open('Email_spam_classifier.pkl', 'rb'))

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Email Spam Classifier", layout="centered")

st.title("📧 Email Spam Classifier")

with st.form(key="email_form"):
    email_text = st.text_area("Enter your email:", height=200, help="Type or paste the email content here")
    submitted = st.form_submit_button("Predict Label")

if submitted:
    
    if not email_text.strip():
        st.warning("⚠️ Please enter an email to classify.")
    
    else:
        query = [email_text]
        
        prediction = pipe.predict(query)[0]
        probability = pipe.predict_proba(query)[0][1]
        
        label = "Spam" if prediction == 1 else "Ham"
        
        st.success(f"**Predicted Label:** {label}")
        st.progress(float(probability))
        st.info(f"**Spam Probability:** {probability:.2f}")
