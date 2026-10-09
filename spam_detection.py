import os
import kagglehub
import pandas as pd
import numpy as np
import joblib
import warnings
import matplotlib.pyplot as plt # 👈 New Import

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Import all models
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings("ignore", category=FutureWarning)

# =====================================================================
# 🎛️ CONFIGURATION: CHOOSE YOUR MAIN MODEL HERE
# =====================================================================
CHOSEN_PROJECT_MODEL = "Linear SVM"
# =====================================================================
# 1. DATA LOADING & CLEANING
# =====================================================================
print("⏳ Downloading and loading dataset...")
path = kagglehub.dataset_download("vishakhdapat/sms-spam-detection-dataset")

csv_files = []
for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith(".csv"):
            csv_files.append(os.path.join(root, file))

if not csv_files:
    raise FileNotFoundError("No CSV file found in the KaggleHub dataset.")

df = pd.read_csv(csv_files[0], encoding='latin-1') 
df = df.iloc[:, :2] 
df.columns = ['label', 'message']

# --- 📊 NEW: CALCULATE & DISPLAY DATASET STRUCTURE ---
total_records = len(df)
ham_count = (df['label'] == 'ham').sum()
spam_count = (df['label'] == 'spam').sum()

# Compute percentages safely
ham_pct = (ham_count / total_records) * 100
spam_pct = (spam_count / total_records) * 100

print("\n=============================================")
print("📊 KAGGLER DATASET METRICS & PROFILE")
print("=============================================")
print(f"Total SMS Messages Sampled : {total_records:,}")
print(f" Legitimate (Ham) Records : {ham_count:,} ({ham_pct:.2f}%)")
print(f" Malicious (Spam) Records  : {spam_count:,} ({spam_pct:.2f}%)")
print("=============================================\n")

# =====================================================================
# 2. PREPROCESSING & VECTORIZATION
# =====================================================================
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label'])

X_train, X_test, y_train, y_test = train_test_split(
    df['message'], 
    df['label_encoded'], 
    test_size=0.2, 
    random_state=42, 
    stratify=df['label_encoded']
)

tfidf = TfidfVectorizer(lowercase=True, stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# =====================================================================
# 3. INITIALIZE ALL MODELS
# =====================================================================
models = {
    "Multinomial Naive Bayes": MultinomialNB(class_prior=[0.5, 0.5]),
    "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=5),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(random_state=42),
    "Linear SVM": CalibratedClassifierCV(SVC(kernel='linear', random_state=42))
}

if CHOSEN_PROJECT_MODEL not in models:
    raise ValueError(f"Invalid model chosen. Choose from: {list(models.keys())}")

# =====================================================================
# 4. TRAIN AND EVALUATE ALL MODELS
# =====================================================================
results = []
print("\n🚀 Training and evaluating all models...")

for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=1)
    recall = recall_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)
    
    results.append({
        "Model": name,
        "Accuracy": f"{accuracy * 100:.2f}%",
        "Spam Precision": f"{precision * 100:.2f}%",
        "Spam Recall": f"{recall * 100:.2f}%",
        "Spam F1-Score": f"{f1 * 100:.2f}%"
    })

# Print performance matrix table
results_df = pd.DataFrame(results)
print("\n--- 📊 MODEL PERFORMANCE COMPARISON ---")
print(results_df.to_string(index=False))

# --- GENERATE GENERATED BAR CHART GRAPH ---
print("\n📊 Generating visualization chart...")
plot_df = results_df.copy()
for col in ["Accuracy", "Spam Precision", "Spam Recall", "Spam F1-Score"]:
    plot_df[col] = plot_df[col].str.rstrip('%').astype('float')

# Plot grouped metrics together side by side
ax = plot_df.plot(x="Model", y=["Accuracy", "Spam Precision", "Spam Recall", "Spam F1-Score"], 
                  kind="bar", figsize=(13, 6), width=0.8, color=['#3b82f6', '#10b981', '#f59e0b', '#ef4444'])

plt.title("SMS Spam Detection - Model Performance Comparison", fontsize=14, fontweight='bold', pad=15)
plt.ylabel("Percentage (%)", fontsize=12)
plt.xlabel("Machine Learning Models", fontsize=12)
plt.xticks(rotation=15, ha="right")
plt.ylim(0, 115) 
plt.grid(axis='y', linestyle='--', alpha=0.4)

# Print percentage text over every single bar element
for container in ax.containers:
    ax.bar_label(container, fmt='%.1f%%', padding=3, fontsize=8)

plt.tight_layout()
plt.savefig("model_comparison.png", dpi=300) # Saved directly to your root folder
plt.show()

# =====================================================================
# 5. LIVE CUSTOM TEXT TESTING FOR ALL MODELS
# =====================================================================
def predict_custom_message(text_message):
    print(f"\n💬 Test Message: \"{text_message}\"")
    transformed_text = tfidf.transform([text_message])
    
    for name, model in models.items():
        prediction = model.predict(transformed_text)[0]
        label_result = le.inverse_transform([prediction])[0].upper()
        
        prob = model.predict_proba(transformed_text)[0]
        spam_prob = prob[1] * 100
        
        marker = "⭐ [YOUR CHOICE]" if name == CHOSEN_PROJECT_MODEL else ""
        print(f" -> {name:25}: [{label_result}] ({spam_prob:.1f}% spam probability) {marker}")

print("\n--- 🔮 Live Testing Across All Models ---")
predict_custom_message("Hey, are we still meeting up for lunch today at 1 PM?")
predict_custom_message("CONGRATULATIONS! You have won a free PS5 gift card. Call 0800123 to claim your prize now!")

# =====================================================================
# 6. SAVE ONLY YOUR SPECIFIC CHOSEN MODEL
# =====================================================================
joblib.dump(models[CHOSEN_PROJECT_MODEL], 'spam_detector_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
joblib.dump(le, 'label_encoder.pkl')

print(f"\n💾 Saved the '{CHOSEN_PROJECT_MODEL}' assets to disk successfully as your primary model!")
