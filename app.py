import gradio as gr
import joblib
import os
import csv
from datetime import datetime

# Load your saved models
model = joblib.load('spam_detector_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')
le = joblib.load('label_encoder.pkl')

def classify_sms(message):
    if not message.strip():
        return "Please enter some text.", "0.00%"
        
    # Transform message to numeric TF-IDF vector
    transformed = tfidf.transform([message])
    
    # Predict the target class
    prediction = model.predict(transformed)
    
    # Safely decode the array prediction into a readable label string
    decoded_label = le.inverse_transform(prediction.ravel())[0]
    label_string = str(decoded_label).upper()
    
    # Calculate target class probability percentage
    prob = model.predict_proba(transformed) * 100
    
    # Extracts the specific probability of the predicted class 
    pred_idx = prediction[0]
    class_probability = prob[0][pred_idx]
    
    return label_string, f"{class_probability:.2f}%"

# New Feature: Function to log misclassified ham messages
def flag_as_ham(message):
    if not message.strip():
        return "⚠️ Cannot flag an empty message."
        
    log_file = "feedback_logs.csv"
    file_exists = os.path.isfile(log_file)
    
    # Append the misclassified text to a CSV file for future retraining
    with open(log_file, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Text", "True_Label", "Predicted_Label"])
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), message, "HAM", "SPAM"])
        
    return "✅ Message logged! This data will be used to retrain the model."

# Custom CSS for styling components
custom_css = """
body { background-color: #f9fafb; }
.container { max-width: 900px; margin: auto; padding-top: 2rem; }
.header { text-align: center; margin-bottom: 2rem; }
.submit-btn { background-color: #2563eb !important; color: white !important; }
.flag-btn { background-color: #ef4444 !important; color: white !important; margin-top: 1rem; }
"""

# Build the improved visual UI using standard blocks
with gr.Blocks(title="SMS Spam Detector") as demo:
    with gr.Group(elem_classes="container"):
        # Application Header
        gr.Markdown(
            """
            #  SMS Spam Detection System
            Analyze incoming text messages in real-time to identify spam or legitimate communications.
            """,
            elem_classes="header"
        )
        
        # Side-by-Side Two-Column Layout
        with gr.Row():
            # Left Column: User Input
            with gr.Column(scale=1):
                gr.Markdown("### ✉️ Input Message")
                input_text = gr.Textbox(
                    lines=5, 
                    placeholder="Type or paste your SMS message here...",
                    label="SMS Text"
                )
                submit_btn = gr.Button("🔍 Analyze Message", elem_classes="submit-btn")
            
            # Right Column: Visual Results & Feedback
            with gr.Column(scale=1):
                gr.Markdown("### Classification Analysis")
                output_label = gr.Label(label="Prediction Result")
                output_prob = gr.Textbox(label="Confidence Level", interactive=False)
                
                # Feedback Interface Elements
                gr.Markdown("### 🛠️ Model Correction")
                flag_btn = gr.Button("🚩 Flag: This is actually HAM", elem_classes="flag-btn")
                feedback_status = gr.Markdown("") 
        
        # Link the button action to our classification function
        submit_btn.click(
            fn=classify_sms,
            inputs=input_text,
            outputs=[output_label, output_prob]
        )
        
        # Link the flag button to the feedback logging function
        flag_btn.click(
            fn=flag_as_ham,
            inputs=input_text,
            outputs=feedback_status
        )

# Launch the app
demo.launch(share=True, css=custom_css)
