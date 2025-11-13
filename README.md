# ML_Mini_Project
Sentiment Analysis Project
This project trains, compares, and selects a "champion" machine learning model for sentiment analysis. It then provides a simple Python script (app.py) to load and run this champion model for real-time predictions.

🚀 How to Run the Final Model (Demo)
This is the simplest way to see the final model in action.

1. Prerequisites
You must have Python installed. You will also need to install the required libraries:

Bash

pip install scikit-learn nltk joblib
2. Get the Model File
The app.py script requires the final, trained model file to be present in the same folder.

Required Filename: Machine_learning_Miniproject.joblib

This .joblib file is the final "champion" model that was generated and saved by the Final_Model_Showdown_(4_Models).ipynb notebook. If your model file is named differently (e.g., best_sentiment_model_final.joblib), you must rename it to Machine_learning_Miniproject.joblib for the app.py script to find it.

3. Run the Application
Execute the app.py script from your terminal:

Bash

python app.py
What to expect:

On the very first run, the script will check for and download necessary NLTK data (like stopwords and wordnet). This only happens once.

You will see a "✅ Model loaded successfully!" message.

You can then type any sentence and press Enter to get a POSITIVE, NEGATIVE, or NEUTRAL sentiment prediction.

Type exit to quit the program.

🏋️‍♂️ How the Model Was Built (Training)
The Final_Model_Showdown_(4_Models).ipynb file contains the complete code used to train and select the final model.

Process Summary
Data Loading: The notebook loads and combines two datasets: Reddit_Data.csv and Twitter_Data.csv.

Note: This notebook was designed to run in Google Colab. It includes code to mount Google Drive (drive.mount('/content/drive')) to access the datasets from a specific path: /content/drive/MyDrive/Dataset/. To re-run the training notebook successfully, you must either place the datasets in that exact Google Drive location or modify the file paths in Block 2 of the notebook.

Preprocessing: All text is cleaned using a standardized function (lowercase, remove URLs/handles, remove punctuation, lemmatize, and remove stopwords). This exact same function is used in app.py to ensure consistency.

Model Showdown: Four different models are trained and compared using a TfidfVectorizer:

Logistic Regression

Random Forest

Gradient Boosting

Naive Bayes

Champion Selection: Based on the test data, Logistic Regression was selected as the "champion" model, achieving the highest accuracy (88.36%).

Saving: This champion model (the Logistic Regression Pipeline) is then saved to a .joblib file in Google Drive, which is the file used by app.py.

📁 File Descriptions
Final_Model_Showdown_(4_Models).ipynb: The Jupyter Notebook used to train, compare, and save the best-performing sentiment analysis model.

app.py: A Python script that loads the final .joblib model and allows you to run live sentiment predictions from your terminal.

Machine_learning_Miniproject.joblib: (This is the expected name) The final, saved Logistic Regression model object, created by the .ipynb notebook.