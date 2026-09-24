import os
import kagglehub
import pandas as pd

# 1. Download the dataset from Kaggle
path = kagglehub.dataset_download("vishakhdapat/sms-spam-detection-dataset")
print("Dataset downloaded to:", path)
print("Files:", os.listdir(path))

# 2. Search for all CSV files in the downloaded folder
csv_files = []
for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith(".csv"):
            csv_files.append(os.path.join(root, file))

print("\nCSV files found:")
for file in csv_files:
    print(file)

# 3. Raise an error if no CSV is found, otherwise load the first one
if not csv_files:
    raise FileNotFoundError("No CSV file found in the KaggleHub dataset.")

csv_path = csv_files[0]
df = pd.read_csv(csv_path)

# --- NEW STEP: Rename columns from v1/v2 to label/message ---
df.columns = ['label', 'message']

# 4. Display dataset structure and preview rows
print("\n--- Dataset Preview ---")
print(df.head())

print("\n--- Metadata & Diagnostics ---")
print(f"Dataset shape: {df.shape}")
print(f"Column names: {df.columns.tolist()}")

print("\nData types:")
print(df.dtypes)

print("\nMissing values count:")
print(df.isnull().sum())

print("\nMissing value percentage:")
print(df.isnull().mean() * 100)

print("\n--- Class Balance (Ham vs Spam count) ---")
print(df['label'].value_counts())
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# 1. Encode target labels ('ham' -> 0, 'spam' -> 1)
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label'])

# 2. Split the dataset into Training (80%) and Testing (20%) sets
# 'stratify' ensures the ham/spam ratio remains identical in both sets
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], 
    df['label_encoded'], 
    test_size=0.2, 
    random_state=42, 
    stratify=df['label_encoded']
)

# 3. Vectorize text using TF-IDF (converts words into numeric weights)
# lowercase=True: Normalizes all text to lowercase
# stop_words='english': Removes common filler words like 'the', 'is', 'at'
tfidf = TfidfVectorizer(lowercase=True, stop_words='english')

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print("--- Preprocessing & Vectorization Complete ---")
print(f"Training features shape: {X_train_tfidf.shape}")
print(f"Testing features shape: {X_test_tfidf.shape}")
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. Initialize and train the Naive Bayes model
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)

# 2. Make predictions on the test dataset
y_pred = nb_model.predict(X_test_tfidf)

# 3. Evaluate the model performance
print("--- Model Performance Evaluation ---")
print(f"Overall Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")

print("--- Classification Report ---")
# 'target_names' matches 0 back to 'ham' and 1 back to 'spam'
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))
import numpy as np

print("\n--- 🛠️ Optimizing Model to Catch More Spam ---")
# Adjust priors to account for class imbalance (helps catch more spam)
nb_optimized = MultinomialNB(class_prior=[0.5, 0.5])
nb_optimized.fit(X_train_tfidf, y_train)

y_pred_opt = nb_optimized.predict(X_test_tfidf)

# Generate report with original text target names to avoid the KeyError
report_dict = classification_report(y_test, y_pred_opt, target_names=le.classes_, output_dict=True)
print(f"New Spam Recall: {report_dict['spam']['recall']:.2f}")

print("New Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_opt))


# --- FUNCTION TO TEST CUSTOM MESSAGES ---
def predict_message(text_message, model=nb_optimized):
    # 1. Transform the input text using the existing TF-IDF vectorizer
    transformed_text = tfidf.transform([text_message])
    
    # 2. Predict the numerical class
    prediction = model.predict(transformed_text)[0]
    
    # 3. Predict the probability percentage for spam (index 1)
    probabilities = model.predict_proba(transformed_text)[0]
    spam_prob = probabilities[1] * 100
    
    # 4. Decode the numeric prediction back to 'ham' or 'spam'
    label_result = le.inverse_transform([prediction])[0]
    
    print(f"\nMessage: \"{text_message}\"")
    print(f"Result:  [{label_result.upper()}] ({spam_prob:.2f}% spam probability)")

# --- TEST THE CUSTOM PREDICTOR ---
print("\n--- 🔮 Testing Custom Messages ---")
predict_message("Hey, are we still meeting up for lunch today at 1 PM?")
predict_message("CONGRATULATIONS! You have won a free PS5 gift card. Call 0800123 to claim your prize now!")
predict_message("Urgent: Your bank account passcode has expired. Please verify your identity here.")
import joblib

# Save the trained model and the vectorizer
joblib.dump(nb_optimized, 'spam_detector_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
joblib.dump(le, 'label_encoder.pkl')
print("\n💾 Model assets saved successfully!")
