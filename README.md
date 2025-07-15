 Spam Email Classifier

A machine learning project that classifies emails as **spam** or **not spam (ham)** using Python and the Naive Bayes algorithm. The model learns from a dataset of labeled emails and predicts whether new emails are spam, helping to filter unwanted messages.

 Project Overview:

Spam emails are unsolicited messages that can be annoying or even dangerous. This project builds a classifier that automatically detects spam emails using text processing and machine learning techniques.

 Tools and Libraries Used

- Python — Programming language used for the project  
- Pandas — For data manipulation and loading the dataset  
- NLTK — Natural Language Toolkit, used here for removing stopwords (common words like "the", "is", etc.)  
- Scikit-learn — Machine learning library used for vectorizing text data and training the Naive Bayes model  


 How the Code Works:

1. Load the Data:  
   The dataset (`spam.csv`) contains emails labeled as "spam" or "ham" (not spam).

2. Clean the Text:
   - Convert all text to lowercase.  
   - Remove punctuation.  
   - Remove common stopwords that do not contribute much to meaning.

3. Label Encoding:  
   Convert the "spam"/"ham" labels into numeric form (spam = 1, ham = 0).

4. Feature Extraction:
   Use `CountVectorizer` to convert cleaned text into a matrix of word counts (bag-of-words).

5. Train-Test Split: 
   Split the dataset into training and testing parts to evaluate performance fairly.

6. Model Training:
   Train a Naive Bayes classifier on the training data.

7. Prediction and Evaluation: 
   Test the model on unseen test data and print accuracy, confusion matrix, and classification report.
 
