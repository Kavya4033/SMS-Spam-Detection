# SMS Spam Detection

A complete, end-to-end Machine Learning pipeline that automatically fetches, preprocesses, and classifies text messages into **Ham** (legitimate) or **Spam** (phishing/fraudulent). The project features a trained **Multinomial Naive Bayes** classifier optimized for class imbalances and includes a live interactive web application interface.

## 📊 Model Performance & Results
Because spam datasets are highly imbalanced, standard classifiers often favor the majority class. By tuning class priors, this model significantly reduced false negatives:

* **Overall Accuracy:** 96.86%
* **Spam Precision:** 1.00 *(Zero false positives; legitimate texts are never accidentally blocked)*
* **Optimized Spam Recall:** 0.94 *(Successfully intercepts 94% of incoming spam/phishing texts)*

### 🔍 Confusion Matrix
```text
[[952  14]  <- [True Ham,  False Spam]
 [  9 140]]  <- [False Ham, True Spam]
```

---

## 🛠️ Project Structure
```text
SMS Spam Detection/
│
├── requirements.txt         # Required Python packages
├── .gitignore               # Excludes virtual environments and model cache
├── spam_detection.py        # Core data science training & optimization script
├── app.py                   # Gradio visual web dashboard script
├── spam_detector_model.pkl  # Trained model weights
├── tfidf_vectorizer.pkl     # Text vectorization mappings
└── label_encoder.pkl        # Numerical target encoder
```

---

##  How to Run the Project

### 1. Installation
Clone the repository, set up your virtual environment, and install the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Train the Model
Run the primary script to download the dataset via KaggleHub, process the text features, evaluate metrics, and save the binary model assets:
```bash
python spam_detection.py
```

### 3. Launch the Web Interface
Boot up the interactive web app to test custom text inputs directly in your browser:
```bash
python app.py
```
Open the local URL generated in your terminal (typically `http://127.0.0.1:7860`).
