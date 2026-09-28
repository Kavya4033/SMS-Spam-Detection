import streamlit as st
import joblib

# Load your saved models
model = joblib.load('spam_detector_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')
le = joblib.load('label_encoder.pkl')

st.set_page_config(page_title="SMS Spam Detector", layout="wide")

st.title("📱 SMS Spam Detection System")
st.write("Analyze incoming text messages in real-time to identify spam or legitimate communications.")

# Side-by-side Layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("✉️ Input Message")
    message = st.text_area("SMS Text", placeholder="Type or paste your SMS message here...", height=150)
    analyze_btn = st.button("🔍 Analyze Message", type="primary")

with col2:
    st.subheader("📊 Classification Analysis")
    if analyze_btn:
        if not message.strip():
            st.warning("Please enter some text.")
        else:
            transformed = tfidf.transform([message])
            prediction = model.predict(transformed)
            
            # Safely decode prediction
            decoded_label = le.inverse_transform(prediction.ravel())
            label_string = str(decoded_label[0]).upper()
            
            # Extract probabilities
            prob = model.predict_proba(transformed) * 100
            class_probability = prob[0][prediction[0]]
            
            # Display beautiful metrics boxes
            st.metric(label="Prediction Result", value=label_string)
            st.metric(label="Confidence Level", value=f"{class_probability:.2f}%")
