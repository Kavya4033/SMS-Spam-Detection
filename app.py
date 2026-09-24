import gradio as gr
import joblib

# Load your saved models
model = joblib.load('spam_detector_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')
le = joblib.load('label_encoder.pkl')

def classify_sms(message):
    transformed = tfidf.transform([message])
    prediction = model.predict(transformed)[0]
    prob = model.predict_proba(transformed)[0][1] * 100
    label = le.inverse_transform([prediction])[0].upper()
    return f"Result: {label} ({prob:.2f}% spam probability)"

# Launch the visual web UI
demo = gr.Interface(
    fn=classify_sms, 
    inputs=gr.Textbox(lines=2, placeholder="Type your SMS message here..."), 
    outputs="text",
    title="SMS Spam Detection"
)
demo.launch(share=True)
