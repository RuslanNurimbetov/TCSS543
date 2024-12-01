import os
import argparse
import re
import torch
import warnings
import xml.etree.ElementTree as ET
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score
from transformers import BertTokenizer, RobertaTokenizer
from transformers import TrainingArguments, Trainer
from datasets import Dataset, DatasetDict

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelBinarizer
from sklearn import metrics

import transformers
from transformers import DistilBertConfig, TFDistilBertModel, DistilBertTokenizer, TFBertForSequenceClassification
from transformers import BertModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow.python.keras import layers, models, Sequential, losses, optimizers
from tensorflow.python.keras.layers import LSTM, Concatenate
from tensorflow.python.keras.callbacks import EarlyStopping

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler

from transformers import BertTokenizer, BertModel
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# prepare to clean the data
import gensim.downloader as api
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
# print(stopwords.words('english'))
REPLACE_BY_SPACE_RE = re.compile(r'[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile(r'[^0-9a-zA-Z\s]')
STOPWORDS = set(stopwords.words('english'))
REPEATED_SPACE_RE = re.compile(r' +')


##### use the clean the text
def clean_text(text):
    text = text.lower()
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    text = BAD_SYMBOLS_RE.sub(' ', text)
    text = re.sub(r'(x)\1+', r'\1', text)
    text = REPEATED_SPACE_RE.sub(' ', text)
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)
    return text

# Prediction function for continuous variable
def predict_agreeable_BERT(train_df, test_df):
    # Split into Train and Test data
    X_train, y_train = train_df['text'], train_df['agr']  # 'agr' is the continuous target variable
    X_test, y_test = test_df['text'], test_df['agr']
    
    # Load BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Tokenize and pad sequences for both train and test data
    train_encodings = tokenizer(X_train.tolist(), padding='max_length', truncation=True, max_length=396, return_tensors='np')
    test_encodings = tokenizer(X_test.tolist(), padding='max_length', truncation=True, max_length=396, return_tensors='np')
    
    # Build the model (adjust for continuous output)
    model = models.Sequential([
        layers.InputLayer(input_shape=(396,)),  # Input layer (adjust for padding)
        layers.Embedding(input_dim=tokenizer.vocab_size, output_dim=64, input_length=396),  # Embedding layer
        layers.LSTM(32, return_sequences=True),  # LSTM layer 1
        layers.LSTM(16),  # LSTM layer 2
        layers.Dense(64, activation='relu'),  # Dense layer with ReLU activation
        layers.Dense(1, activation='linear')  # Output layer with linear activation (for regression)
    ])
    
    # Compile the model with Mean Squared Error loss (appropriate for regression)
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
    
    # Train the model
    model.fit(train_encodings['input_ids'], y_train, epochs=5, batch_size=64)

    # Make predictions on the test set
    y_pred = model.predict(test_encodings['input_ids'])
    test_df['agr_predicted'] = y_pred

    # Evaluate the predictions
    # mse = metrics.mean_squared_error(y_test, y_pred)
    # rmse = np.sqrt(mse)
    # print(f'Mean Squared Error: {mse:.4f}')
    # print(f'Root Mean Squared Error: {rmse:.4f}')


# Prediction function for continuous variable
def predict_conscientious_BERT(train_df, test_df):
    # Split into Train and Test data
    X_train, y_train = train_df['text'], train_df['con']  # 'con' is the continuous target variable
    X_test, y_test = test_df['text'], test_df['con']
    
    # Load BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Tokenize and pad sequences for both train and test data
    train_encodings = tokenizer(X_train.tolist(), padding='max_length', truncation=True, max_length=396, return_tensors='np')
    test_encodings = tokenizer(X_test.tolist(), padding='max_length', truncation=True, max_length=396, return_tensors='np')
    
    # Build the model (adjust for continuous output)
    model = models.Sequential([
        layers.InputLayer(input_shape=(396,)),  # Input layer (adjust for padding)
        layers.Embedding(input_dim=tokenizer.vocab_size, output_dim=64, input_length=396),  # Embedding layer
        layers.LSTM(32, return_sequences=True),  # LSTM layer 1
        layers.LSTM(16),  # LSTM layer 2
        layers.Dense(64, activation='relu'),  # Dense layer with ReLU activation
        layers.Dense(1, activation='linear')  # Output layer with linear activation (for regression)
    ])
    
    # Compile the model with Mean Squared Error loss (appropriate for regression)
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
    
    # Train the model
    model.fit(train_encodings['input_ids'], y_train, epochs=5, batch_size=64)

    # Make predictions on the test set
    y_pred = model.predict(test_encodings['input_ids'])
    test_df['con_predicted'] = y_pred 

    # Evaluate the predictions
    # mse = metrics.mean_squared_error(y_test, y_pred)
    # rmse = np.sqrt(mse)
    # print(f'Mean Squared Error: {mse:.4f}')
    # print(f'Root Mean Squared Error: {rmse:.4f}')

# Prediction function for continuous variable
def predict_extrovert_BERT(train_df, test_df):
    # Split into Train and Test data
    X_train, y_train = train_df['text'], train_df['ext']  # 'ext' is the continuous target variable
    X_test, y_test = test_df['text'], test_df['ext']
    
    # Load BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Tokenize and pad sequences for both train and test data
    train_encodings = tokenizer(X_train.tolist(), padding='max_length', truncation=True, max_length=396, return_tensors='np')
    test_encodings = tokenizer(X_test.tolist(), padding='max_length', truncation=True, max_length=396, return_tensors='np')
    
    # Build the model (adjust for continuous output)
    model = models.Sequential([
        layers.InputLayer(input_shape=(396,)),  # Input layer (adjust for padding)
        layers.Embedding(input_dim=tokenizer.vocab_size, output_dim=64, input_length=396),  # Embedding layer
        layers.LSTM(32, return_sequences=True),  # LSTM layer 1
        layers.LSTM(16),  # LSTM layer 2
        layers.Dense(64, activation='relu'),  # Dense layer with ReLU activation
        layers.Dense(1, activation='linear')  # Output layer with linear activation (for regression)
    ])
    
    # Compile the model with Mean Squared Error loss (appropriate for regression)
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
    
    # Train the model
    model.fit(train_encodings['input_ids'], y_train, epochs=5, batch_size=64)

    # Make predictions on the test set
    y_pred = model.predict(test_encodings['input_ids'])
    test_df['ext_predicted'] = y_pred

    # Evaluate the predictions
    # mse = metrics.mean_squared_error(y_test, y_pred)
    # rmse = np.sqrt(mse)
    # print(f'Mean Squared Error: {mse:.4f}')
    # print(f'Root Mean Squared Error: {rmse:.4f}')


# Prediction function for continuous variable (neuroticism)
def predict_neurotic_BERT(train_df, test_df):
    # Split into Train and Test data
    X_train, y_train = train_df['text'], train_df['neu']  # 'neu' is the continuous target variable
    X_test, y_test = test_df['text'], test_df['neu']
    
    # Load BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Tokenize and pad sequences for both train and test data
    train_encodings = tokenizer(X_train.tolist(), padding='max_length', truncation=True, max_length=396, return_tensors='np')
    test_encodings = tokenizer(X_test.tolist(), padding='max_length', truncation=True, max_length=396, return_tensors='np')
    
    # Build the model (adjusted for simplicity)
    model = models.Sequential([
        layers.InputLayer(input_shape=(396,)),
        layers.Embedding(input_dim=30522, output_dim=32, input_length=396),  # Adjust the vocab size and dimensions
        layers.LSTM(32, return_sequences=False),
        layers.Dense(16, activation='relu', kernel_initializer='he_normal'),
        # Output layer will vary based on the activation method
        layers.Dense(1, activation='sigmoid'),  # Sigmoid output is in [0, 1]
        layers.Lambda(lambda x: x * 5)
    ])
    
    # Compile the model with Mean Squared Error loss (appropriate for regression)
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
    
    # Train the model
    model.fit(train_encodings['input_ids'], y_train, epochs=10, batch_size=32)

    # Make predictions on the test set
    y_pred = model.predict(test_encodings['input_ids'])
    test_df['neu_predicted'] = y_pred

    # Evaluate the predictions
    # mse = metrics.mean_squared_error(y_test, y_pred)
    # rmse = np.sqrt(mse)
    # print(f'Mean Squared Error: {mse:.4f}')
    # print(f'Root Mean Squared Error: {rmse:.4f}')


# Prediction function for continuous variable
# Prediction function
# Prediction function
def predict_open_BERT(train_df, test_df):
    # Split into Train and Test data
    X_train, y_train = train_df['text'], train_df['ope']
    X_test, y_test = test_df['text'], test_df['ope']
    
    # Load BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Tokenize and pad sequences
    train_encodings = tokenizer(X_train.tolist(), padding='max_length', truncation=True, max_length=396, return_tensors='np')
    test_encodings = tokenizer(X_test.tolist(), padding='max_length', truncation=True, max_length=396, return_tensors='np')
    
    # Scale y_train to range [0, 1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    y_train_scaled = scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    
    # Model
    model = models.Sequential([
        layers.InputLayer(input_shape=(396,)),
        layers.Embedding(input_dim=tokenizer.vocab_size, output_dim=64, input_length=396),
        layers.LSTM(32, return_sequences=True),
        layers.LSTM(16),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # For normalized output (0 to 1)
    ])
    
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
    
    # Train model
    model.fit(train_encodings['input_ids'], y_train_scaled, epochs=5, batch_size=64, validation_split=0.2)

    # Predict and scale back predictions to original range
    y_pred_scaled = model.predict(test_encodings['input_ids'])
    y_pred = scaler.inverse_transform(y_pred_scaled).flatten()
    
    # Explicitly assign predictions to DataFrame using .loc to avoid copy warnings
    test_df.loc[:, 'ope_predicted'] = y_pred
    
    return test_df

    # Evaluate the predictions
    # mse = metrics.mean_squared_error(y_test, y_pred)
    # rmse = np.sqrt(mse)
    # print(f'Mean Squared Error: {mse:.4f}')
    # print(f'Root Mean Squared Error: {rmse:.4f}')

def predict_age_clean_text(train_df, test_df):
    # Training a Naive Bayes model
    count_vect = CountVectorizer(stop_words='english')
    tfid_vect = TfidfVectorizer()
    X_train = count_vect.fit_transform(train_df['text'])
    # X_train = tfid_vect.fit_transform(train_df['text'])
    #test_df['age_range'] = test_df['age'].apply(lambda value: 'xx-24' if value < 25  else ('25-34' if 25 <= value <= 34 else ('35-49' if 35<= value <= 49 else '50-xx')))

    y_train = train_df['age'].apply(lambda value: 'xx-24' if value < 25  else ('25-34' if 25 <= value <= 34 else ('35-49' if 35<= value <= 49 else '50-xx')))
    clf = MultinomialNB(alpha=0.7)
    clf.fit(X_train, y_train)

    # Testing the Naive Bayes model
    X_test = count_vect.transform(test_df['text'])
    # X_test = tfid_vect.transform(test_df['text'])
    # Make predictions on the test data
    y_test_predicted = clf.predict(X_test)

    # Add the predicted age to the test dataframe using lambda function
    test_df['age_predicted'] = y_test_predicted

def predict_gender_clean_text2(train_df, test_df):
    # Split into Train and Test data
    X_train = train_df['text']
    y_train = train_df['gender']
    X_test = test_df['text']
    y_test = test_df['gender']
    
    # TF-IDF Vectorizer with n-grams and max features
    vectorizer_tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1, 3), stop_words='english')
    
    # Classifiers to test
    classifiers = [
        ('Logistic Regression', LogisticRegression(C=1.0, solver='liblinear')),
        ('Random Forest', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('SVM', SVC(kernel='linear', class_weight='balanced')),
        ('Naive Bayes', MultinomialNB())
    ]
    
    # Iterate through classifiers and evaluate
    for clf_name, clf in classifiers:
        print(f"Training {clf_name}...")
        
        # Create the pipeline
        model = Pipeline([('vectorizer', vectorizer_tfidf), ('classifier', clf)])
        
        # Cross-validation to evaluate the classifier
        scores = cross_val_score(model, X_train, y_train, cv=5)
        print(f"{clf_name} Cross-validation score: {scores.mean():.4f}")
        
        # Fit the model to the training data
        model.fit(X_train, y_train)
        
        # Predict on the test data
        predicted_test = model.predict(X_test)
        
        # Add the predicted gender to the test dataframe
        test_df['gender_predicted'] = predicted_test
        test_df['gender_predicted'] = test_df['gender_predicted'].apply(lambda value: 'male' if value == 0 else 'female')
        
        # Evaluate using classification report
        print(f"{clf_name} Classification Report:\n", classification_report(y_test, predicted_test))

    
def predict_gender_clean_text(train_df, test_df):
    # Split into Train and Test data
    X_train = train_df['text']
    y_train = train_df['gender']
    X_test = test_df['text']
    y_test = test_df['gender']
    
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=None, stratify=train_df['gender'])

    # TF-IDF Model
    # Initizalize the vectorizer with max nr words and ngrams (1: single words, 2: two words in a row)
    vectorizer_tfidf = TfidfVectorizer(max_features=30000, ngram_range=(1,2))
    # Fit the vectorizer to the training data
    vectorizer_tfidf.fit(X_train)

    classifier_tfidf = LogisticRegression()
    model_tfidf = Pipeline([("vectorizer", vectorizer_tfidf), ("classifier", classifier_tfidf)])

    model_tfidf.fit(X_train, y_train)
    predicted_test_tfidf = model_tfidf.predict(X_test)

    # Add the predicted gender to the test dataframe using lambda function
    test_df['gender_predicted'] = predicted_test_tfidf
    test_df['gender_predicted'] = test_df['gender_predicted'].apply(lambda value: 'male' if value == 0 else 'female')



def model_test_evaluation(train_df):
    # Splitting the data into 8000 training instances and 1500 test instances
    n = 1500
    test_df = train_df.iloc[:n]
    train_df = train_df.iloc[n:]

    # #### test gender prediction model
    # gender_data_test_df = data_test_df.copy()
    # gender_data_train_df = data_train_df.copy()
    # predict_gender_clean_text2(gender_data_train_df, gender_data_test_df)
    # # print(gender_data_test_df['gender_predicted'])

    # y_gender_test = gender_data_test_df['gender'].apply(lambda value: 'male' if value == 0 else 'female')
    # y_gender_predicted = gender_data_test_df['gender_predicted']
    # # Reporting on classification performance
    # print("# Gender Prediction")
    # print("Accuracy: %.2f" % accuracy_score(y_gender_test,y_gender_predicted))
    # classes = ["male","female"]
    # cnf_matrix = confusion_matrix(y_gender_test,y_gender_predicted,labels=classes)
    # print("Confusion matrix:")
    # print(cnf_matrix)

    # #### test gender prediction model
    # age_data_test_df = data_test_df.copy()
    # age_data_train_df = data_train_df.copy()
    # predict_age_clean_text(age_data_train_df, age_data_test_df)
    # # print(gender_data_test_df['gender_predicted'])

    # #### test open prediction model
    # ope_data_test_df = data_test_df.copy()
    # ope_data_train_df = data_train_df.copy()
    # # predict_open_BERT(ope_data_train_df, ope_data_test_df)
    # predict_agreeable_BERT(ope_data_train_df, ope_data_test_df)
    # predict_open_BERT(train_df, test_df)
    # print(test_df['ope_predicted'])
    # predict_neurotic_BERT(train_df, test_df)
    # print(test_df['neu_predicted'])
    # predict_conscientious_BERT(train_df, test_df)
    # print(test_df['con_predicted'])
    # predict_agreeable_BERT(train_df, test_df)
    # print(test_df['agr_predicted'])
    # predict_extrovert_BERT(train_df, test_df)
    # print(test_df['ext_predicted'])

    

##### Function to read the text/comments #####
def read_text(file_dir, userid):
    file_path = os.path.join(file_dir, f'{userid}.txt')
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
            text = file.read()
            return text
    else:
        return ""
    
##### Generate the output with parsing arguments from the command #####
def save_to_XML_file(input_dir, output_dir):
    # Data path
    training_files_path = 'data/training/profile/profile.csv'
    text_files_dir = 'data/training/text/'
    # Read CSV files
    train_df = pd.read_csv(training_files_path)
    test_df = pd.read_csv(os.path.join(input_dir, 'profile/profile.csv'))

    # Add text/comments into the 'text' column, match with userid using lambda function
    train_df['text'] = train_df['userid'].apply(lambda userid: read_text(text_files_dir, userid))
    test_df['text'] = test_df['userid'].apply(lambda userid: read_text(os.path.join(input_dir, 'text/'), userid))

    train_df['text'] = train_df['text'].apply(clean_text)
    train_df['text'] = train_df['text'].replace('\d+', '')
    test_df['text'] = test_df['text'].apply(clean_text)
    test_df['text'] = test_df['text'].replace('\d+', '')

    # model_test_evaluation(train_df)
    predict_gender_clean_text(train_df, test_df)
    predict_age_clean_text(train_df, test_df)
    predict_open_BERT(train_df, test_df)
    predict_neurotic_BERT(train_df, test_df)
    predict_conscientious_BERT(train_df, test_df)
    predict_agreeable_BERT(train_df, test_df)
    predict_extrovert_BERT(train_df, test_df)

    for index, row in test_df.iterrows():
        user = ET.Element("user")
        
        user.set("id", str(row['userid']))
        user.set("age_group", str(row['age_predicted']))
        user.set("gender", str(row['gender_predicted']))
        user.set("extrovert", str(row['ext_predicted']))
        user.set("neurotic", str(row['neu_predicted']))
        user.set("agreeable", str(row['agr_predicted']))
        user.set("conscientious", str(row['con_predicted']))
        user.set("open", str(row['ope_predicted']))
    
        file_name = f"{row['userid']}.xml"
        tree = ET.ElementTree(user)
        tree.write(os.path.join(output_dir, file_name), encoding='utf-8', xml_declaration=True)
    
    print(f"All the XML files for each userId have been created in the '{output_dir}' directory.")


##### Main method to generate the output with parsing arguments from the command #####
def main():
    parser = argparse.ArgumentParser(description='Check input and output directory paths.')
    parser.add_argument('-i', '--input', required=True, help='Path to the input directory')
    parser.add_argument('-o', '--output', required=True, help='Path to the output directory')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    # os.makedirs(args.output, exist_ok=True)

    # Call the function to create XML files
    save_to_XML_file(args.input, args.output)

if __name__ == '__main__':
    main()