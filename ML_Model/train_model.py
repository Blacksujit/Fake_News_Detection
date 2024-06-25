import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import os
# ------------------------------------------OLD DATASET ---------------------------------------###############
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# # Load the dataset
data = pd.read_csv('C:\\Users\\HP\\OneDrive\\Desktop\\Machine Learning Projects\\Fake_News_Detection\\data\\news.csv')

# print(data.head(3))
# --------------------------------------------------# NEW CODE   #######################  

# New dataset Loading 

# # Load the true and false datasets
# true_data = pd.read_csv('C:\\Users\\HP\\OneDrive\\Desktop\\Mchine Learning Projects\\Fake_News_Detection\\data\\True.csv')
# false_data = pd.read_csv('C:\\Users\\HP\\OneDrive\\Desktop\\Mchine Learning Projects\\Fake_News_Detection\\data\\Fake.csv')


# # Combine Two dataset
# # Combine the datasets
# data = pd.concat([true_data, false_data], ignore_index=True)

# # Shuffle the data
# data = data.sample(frac=1).reset_index(drop=True)

# print(data.head(3))

# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer

# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')

# # ------------------------------------------# Preprocess text---------------------------------------------------------------------------------
# def preprocess(text):
#     lemmatizer = WordNetLemmatizer()
#     tokens = word_tokenize(text.lower())
#     tokens = [word for word in tokens if word.isalpha()]
#     tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords.words('english')]
#     return ' '.join(tokens)

# data['clean_text'] = data['text'].apply(preprocess)


# # -----------------------# Feature Extraction and Model Training--------------------


# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, classification_report
# import joblib
# import os

# # Feature extraction using TF-IDF with bigrams
# vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
# X = vectorizer.fit_transform(data['clean_text'])
# y = data['label']

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Hyperparameter tuning using Grid Search
# param_grid = {
#     'C': [0.01, 0.1, 1, 10, 100],
#     'penalty': ['l1', 'l2'],
#     'solver': ['liblinear']  # 'liblinear' supports both 'l1' and 'l2' penalties
# }

# log_reg = LogisticRegression()

# grid_search = GridSearchCV(log_reg, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
# grid_search.fit(X_train, y_train)

# # Get the best parameters and model
# best_params = grid_search.best_params_
# best_model = grid_search.best_estimator_

# print("Best Parameters:", best_params)
# print("Best Model Accuracy on Training Data:", best_model.score(X_train, y_train) * 100)
# print("Best Model Accuracy on Test Data:", best_model.score(X_test, y_test) * 100)

# # Evaluate the best model
# y_pred = best_model.predict(X_test)
# print('Accuracy:', accuracy_score(y_test, y_pred) * 100)
# print(classification_report(y_test, y_pred))

# # Ensure the models directory exists
# model_dir = os.path.join('..', 'models')
# os.makedirs(model_dir, exist_ok=True)

# # Save the best model and vectorizer
# joblib.dump(best_model, os.path.join(model_dir, 'fake_news_detector.pkl'))
# joblib.dump(vectorizer, os.path.join(model_dir, 'vectorizer.pkl'))



# # ---------------------------------------------------------------------------------###################



# # Add labels: 1 for true, 0 for false
# true_data['label'] = 1
# false_data['label'] = 0


# ------------------------------------------------------------# OLD CODE# -------------------------------------------------------



print(data.head(3))

# Preprocess text
def preprocess(text):
    lemmatizer = WordNetLemmatizer()
    # tokens = word_tokenize(text.lower())
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

# data['clean_text'] = data['text'].apply(preprocess)
data['clean_text'] = data['text'].apply(preprocess)


# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000 , ngram_range=(1,2))
X = vectorizer.fit_transform(data['clean_text'])
y = data['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning using Grid Search
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']  # 'liblinear' supports both 'l1' and 'l2' penalties
}

log_reg = LogisticRegression()

grid_search = GridSearchCV(log_reg, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Get the best parameters and model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print("Best Parameters:", best_params)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)
print(model.score(X_train , y_train)*100)
print(model.score(X_test , y_test)*100)

# Evaluate the model
y_pred = model.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred)*100)
print(classification_report(y_test, y_pred))



# Ensure the models directory exists
model_dir = os.path.join('..', 'ML_Model')
os.makedirs(model_dir, exist_ok=True)


# Save the model and vectorizer
joblib.dump(model, os.path.join(model_dir, 'fake_news_detector.pkl'))
joblib.dump(vectorizer, os.path.join(model_dir, 'vectorizer.pkl'))




