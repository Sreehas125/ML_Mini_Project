import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk

# --- Download NLTK data locally (run this part once) ---
# This checks if the data exists and downloads it if needed.
try:
    stopwords.words('english')
except LookupError:
    print("Downloading NLTK data... (This happens only once)")
    # Using 'all' is a robust way to prevent lookup errors.
    nltk.download('all')

# --- The exact same cleaning function from your training script ---
# It's crucial that preprocessing is identical for training and testing.
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    tokens = word_tokenize(text)
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(cleaned_tokens)

# --- Load your champion model from the file ---
MODEL_PATH = "Machine_learning_Miniproject.joblib"
print(f"Loading the champion model from: {MODEL_PATH}")
try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully!")
except FileNotFoundError:
    print(f"ERROR: Model file not found at '{MODEL_PATH}'")
    print("Please make sure the model file is in the same folder as this script.")
    exit()


# --- Main loop to test the model interactively ---
if __name__ == "__main__":
    print("\nSentiment Analysis Demo (Final Champion Model). Type 'exit' to quit.")
    while True:
        # Get input from the user
        user_input = input("Enter a sentence to analyze: ")
        if user_input.lower() == 'exit':
            break

        # Clean the input text using the same function
        cleaned_input = clean_text(user_input)

        # Make a prediction using the loaded model
        prediction = model.predict([cleaned_input])

        # Print the result
        print(f"--> PREDICTION: {prediction[0].upper()}\n")

