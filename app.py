import re
import joblib
import gradio as gr

# Load saved assets
model = joblib.load('spam_detector_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')
le = joblib.load('label_encoder.pkl')

def analyze_sms(message):
    if not message.strip():
        return (
            "<div style='padding:15px; border-radius:8px; background-color:#f3f4f6; color:#1f2937; font-weight:bold; text-align:center;'>⚠️ Please enter some text to analyze.</div>",
            "0.00%", 
            "N/A", 
            "N/A", 
            "📂 Status: **Awaiting Input**"
        )
        
    # 1. Base ML Model Predictions
    transformed = tfidf.transform([message])
    prediction = model.predict(transformed)
    
    # Safely extract values handles multi-model variations smoothly
    pred_idx = int(prediction[0])
    decoded_label = le.inverse_transform([pred_idx])
    label_string = str(decoded_label[0]).upper()
    
    # Handle matrix configurations cleanly
    prob = model.predict_proba(transformed)
    # Extract the probability of the Spam class (index 1)
    spam_prob = prob[0][1] * 100 
    
    # 2. Advanced Feature Insights (Heuristic checks)
    has_url = "⚠️ Yes" if re.search(r'http\S+|www\S+|https\S+', message) else "✅ None detected"
    has_phone = "⚠️ Yes" if re.search(r'\b\d{7,15}\b', message) else "✅ None detected"
    
    # Replacement for Character Counter: Text Status Label
    if label_string == "SPAM":
        status_display = "🚨 Message Status: **SPAM (Dangerous)**"
    else:
        status_display = "🛡️ Message Status: **HAM (Safe)**"

    # 3. Dynamic Visual Alert HTML Generator
        # 3. Dynamic Visual Alert HTML Generator
    if label_string == "SPAM":
        alert_html = f"""
        <div style="padding: 20px; border-radius: 12px; background-color: #fee2e2; border: 2px solid #ef4444; text-align: center; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);">
            <span style="font-size: 2rem; display: block; margin-bottom: 5px;">🚨</span>
            <strong style="font-size: 1.4rem; letter-spacing: 0.5px; color: #000000; display: block;">SECURITY WARNING: SPAM DETECTED</strong>
            <p style="margin-top: 8px; font-size: 1rem; color: #000000;">This message displays high-risk structural matches typically linked to phishing, fraud, or advertisement scams.</p>
        </div>
        """

    else:
        alert_html = f"""
        <div style="padding: 20px; border-radius: 12px; background-color: #dcfce7; border: 2px solid #22c55e; color: #166534; text-align: center; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);">
            <span style="font-size: 2rem; display: block; margin-bottom: 5px;">🛡️</span>
            <strong style="font-size: 1.4rem; tracking: 0.5px;">SECURE & LEGITIMATE (HAM)</strong>
            <p style="margin-top: 8px; font-size: 1rem; color: #14532d;">This text looks safe. It resembles natural conversational patterns and displays standard verification metrics.</p>
        </div>
        """

    return alert_html, f"{spam_prob:.2f}%", has_url, has_phone, status_display

# --- Interactive Examples Library ---
example_templates = [
    ["Hey, are we still meeting up for lunch today at 1 PM?"],
    ["CONGRATULATIONS! You have won a free PS5 gift card. Call 0800123 to claim your prize now!"],
    ["Urgent: Your bank account passcode has expired. Please verify your identity here."],
    ["Can you please pick up some milk on your way home from work? Thanks!"]
]

# --- Custom UI Stylesheet injection ---
custom_css = """
footer { visibility: hidden !important; }
.title-banner { background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%); padding: 25px; border-radius: 12px; color: white; text-align: center; margin-bottom: 25px; }
.meta-box { background-color: var(--background-fill-secondary); border-radius: 8px; padding: 12px; text-align: center; font-size: 1.05rem; font-weight: 600; }
"""

# Assemble the UI Layout Structure
with gr.Blocks(title="SMS Spam Detection Dashboard") as demo:
    
    # Header Banner (Removed "Firewall")
    gr.HTML(
        """
        <div class="title-banner">
            <h1 style="color: white; margin: 0; font-size: 2.2rem; font-weight: 800;">🛡️ SMS Spam Detection System</h1>
            <p style="color: #dbeafe; margin: 8px 0 0 0; font-size: 1.1rem;">Enterprise NLP security analysis engine for incoming real-time messages</p>
        </div>
        """
    )
    
    with gr.Row():
        # Left Workspace: Core Processing
        with gr.Column(scale=3):
            gr.Markdown("### 📥 Input Core")
            input_text = gr.Textbox(
                lines=6, 
                placeholder="Type, paste, or select a sample SMS message from below to initiate analysis...",
                label="Message Content Pipeline"
            )
            
            # Action controls (Removed "Firewall")
            with gr.Row():
                clear_btn = gr.Button("♻️ Clear Text", variant="secondary")
                submit_btn = gr.Button("🔍 Analyze Message Log", variant="primary")
            
            # Injected Clickable Preset Examples
            gr.Examples(
                examples=example_templates,
                inputs=input_text,
                label="💡 Quick Test Templates"
            )

        # Right Workspace: Deep Insights Reporting Matrix
        with gr.Column(scale=2):
            gr.Markdown("### 📊 Live Diagnostic Insights")
            
            # HTML Security Status Output Block
            output_html = gr.HTML(value="<div style='text-align:center; padding:20px; color:var(--text-color-subdued);'>Awaiting engine execution query...</div>")
            
            # Replaced Character/Word count box with the new structural label status
            output_meta = gr.Markdown(value="📂 Status: **Awaiting Input**", elem_classes="meta-box")
            
            # Risk Breakdown metrics wrapped inside a proper Layout Group container
            with gr.Group():
                gr.Markdown("#### Risk Assessment Matrix")
                output_prob = gr.Textbox(label="Evaluated Spam Probability Score", interactive=False)
                
            with gr.Accordion("🔍 Structural Risk Insights", open=True):
                output_url = gr.Textbox(label="Contains URL/Hyperlinks", interactive=False)
                output_phone = gr.Textbox(label="Contains Phone Number Sequences", interactive=False)

    # Establish Reactive Bindings
    submit_btn.click(
        fn=analyze_sms,
        inputs=input_text,
        outputs=[output_html, output_prob, output_url, output_phone, output_meta]
    )
    
    # Connect standard clear reset mechanism
    clear_btn.click(
        fn=lambda: ("", "<div style='text-align:center; padding:20px; color:var(--text-color-subdued);'>Awaiting engine execution query...</div>", "0.00%", "N/A", "N/A", "📂 Status: **Awaiting Input**"),
        inputs=None,
        outputs=[input_text, output_html, output_prob, output_url, output_phone, output_meta]
    )

# Launch local thread pipeline with styling parameters
if __name__ == "__main__":
    demo.launch(share=True, theme=gr.themes.Soft(), css=custom_css)
