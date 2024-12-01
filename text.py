#!/usr/bin/python3

import pandas as pd
import os
import xml.etree.ElementTree as ET
import argparse
import numpy as np
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import KFold, cross_val_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import ComplementNB

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

##### Linear regression to predict agreeable #####
def predict_agreeable_LR(train_df, test_df):
    # convert text to numerical data
    tfid_vect = TfidfVectorizer()
    # X_train = count_vect.fit_transform(train_df['text'])
    X_train = tfid_vect.fit_transform(train_df['text'])
    y_train = train_df['agr']
    slRegressor = LinearRegression()
    slRegressor.fit(X_train, y_train)
    # Testing theLinear Regression model
    X_test = tfid_vect.transform(test_df['text'])
    # Make predictions on the test data
    y_test_predicted = slRegressor.predict(X_test)
    # print(y_test_predicted)

    # Add the predicted age to the test dataframe using lambda function
    test_df['agr_predicted'] = y_test_predicted
    # print(test_df['agr_predicted'])

##### Linear regression to predict conscientious #####
def predict_conscientious_LR(train_df, test_df):
    # convert text to numerical data
    tfid_vect = TfidfVectorizer()
    # X_train = count_vect.fit_transform(train_df['text'])
    X_train = tfid_vect.fit_transform(train_df['text'])
    y_train = train_df['con']
    slRegressor = LinearRegression()
    slRegressor.fit(X_train, y_train)
    # Testing theLinear Regression model
    X_test = tfid_vect.transform(test_df['text'])
    # Make predictions on the test data
    y_test_predicted = slRegressor.predict(X_test)
    # print(y_test_predicted)

    # Add the predicted age to the test dataframe using lambda function
    test_df['con_predicted'] = y_test_predicted
    # print(test_df['con_predicted'])

##### Linear regression to predict extrovert #####
def predict_extrovert_LR(train_df, test_df):
    # convert text to numerical data
    tfid_vect = TfidfVectorizer()
    # X_train = count_vect.fit_transform(train_df['text'])
    X_train = tfid_vect.fit_transform(train_df['text'])
    y_train = train_df['ext']
    slRegressor = LinearRegression()
    slRegressor.fit(X_train, y_train)
    # Testing theLinear Regression model
    X_test = tfid_vect.transform(test_df['text'])
    # Make predictions on the test data
    y_test_predicted = slRegressor.predict(X_test)
    # print(y_test_predicted)

    # Add the predicted age to the test dataframe using lambda function
    test_df['ext_predicted'] = y_test_predicted
    # print(test_df['ext_predicted'])

##### Linear regression to predict neurotic #####
def predict_neurotic_LR(train_df, test_df):
    # convert text to numerical data
    tfid_vect = TfidfVectorizer()
    # X_train = count_vect.fit_transform(train_df['text'])
    X_train = tfid_vect.fit_transform(train_df['text'])
    y_train = train_df['neu']
    slRegressor = LinearRegression()
    slRegressor.fit(X_train, y_train)
    # Testing theLinear Regression model
    X_test = tfid_vect.transform(test_df['text'])
    # Make predictions on the test data
    y_test_predicted = slRegressor.predict(X_test)
    # print(y_test_predicted)

    # Add the predicted age to the test dataframe using lambda function
    test_df['neu_predicted'] = y_test_predicted
    # print(test_df['neu_predicted'])

##### Linear regression to predict the open #####
def predict_open_LR(train_df, test_df):
    # convert text to numerical data
    tfid_vect = TfidfVectorizer()
    # X_train = count_vect.fit_transform(train_df['text'])
    X_train = tfid_vect.fit_transform(train_df['text'])
    y_train = train_df['ope']
    slRegressor = LinearRegression()
    slRegressor.fit(X_train, y_train)
    # Testing theLinear Regression model
    X_test = tfid_vect.transform(test_df['text'])
    # Make predictions on the test data
    y_test_predicted = slRegressor.predict(X_test)
    # print(y_test_predicted)

    # Add the predicted age to the test dataframe using lambda function
    test_df['ope_predicted'] = y_test_predicted
    # print(test_df['ope_predicted'])

##### Naive Bayes model for age recognition #####
def predict_age_NB(train_df, test_df):
    # Training a Naive Bayes model
    count_vect = CountVectorizer(stop_words='english')
    tfid_vect = TfidfVectorizer()
    # X_train = count_vect.fit_transform(train_df['text'])
    X_train = tfid_vect.fit_transform(train_df['text'])
    #test_df['age_range'] = test_df['age'].apply(lambda value: 'xx-24' if value < 25  else ('25-34' if 25 <= value <= 34 else ('35-49' if 35<= value <= 49 else '50-xx')))

    y_train = train_df['age'].apply(lambda value: 'xx-24' if value < 25  else ('25-34' if 25 <= value <= 34 else ('35-49' if 35<= value <= 49 else '50-xx')))
    clf = MultinomialNB(alpha=0.7)
    clf.fit(X_train, y_train)

    # Testing the Naive Bayes model
    # X_test = count_vect.transform(test_df['text'])
    X_test = tfid_vect.transform(test_df['text'])
    # Make predictions on the test data
    y_test_predicted = clf.predict(X_test)

    # Add the predicted age to the test dataframe using lambda function
    test_df['age_predicted'] = y_test_predicted
    # test_df['age_predicted'] = test_df['age_predicted'].apply(lambda value: 'xx-24' if value < 25  else ('25-34' if 25 <= value <= 34 else ('35-49' if 35<= value <= 49 else '50-xx')))

    # Set up k-fold cross-validation
    k = KFold(n_splits=10, shuffle=False, random_state=None)
    cv_scores = cross_val_score(clf, X_train, y_train, cv=k, scoring='accuracy')

    # Calculate and print the mean accuracy
    # print(f"K-Fold Cross-Validation Accuracy Scores: {cv_scores}")
    # print(f"Age Prediction Mean Accuracy: {cv_scores.mean() * 100:.2f}%")

##### Naive Bayes model for gender recognition #####
def predict_gender_NB(train_df, test_df):
    # Training a Naive Bayes model
    count_vect = CountVectorizer()
    tfid_vect = TfidfVectorizer(stop_words='english')
    X_train = count_vect.fit_transform(train_df['text'])
    # X_train = tfid_vect.fit_transform(train_df['text'])
    y_train = train_df['gender']
    clf = MultinomialNB(alpha=0.3)
    clf.fit(X_train, y_train)

    # Testing the Naive Bayes model
    X_test = count_vect.transform(test_df['text'])
    # X_test = tfid_vect.transform(test_df['text'])
    # Make predictions on the test data
    y_test_predicted = clf.predict(X_test)

    # Add the predicted gender to the test dataframe using lambda function
    test_df['gender_predicted'] = y_test_predicted
    test_df['gender_predicted'] = test_df['gender_predicted'].apply(lambda value: 'male' if value == 0 else 'female')


    # Set up k-fold cross-validation
    k = KFold(n_splits=10, shuffle=False, random_state=None)
    cv_scores = cross_val_score(clf, X_train, y_train, cv=k, scoring='accuracy')

    # Calculate and print the mean accuracy
    # print(f"K-Fold Cross-Validation Accuracy Scores: {cv_scores}")
    # print(f"Gender Prediction Mean Accuracy: {cv_scores.mean() * 100:.2f}%")

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

    # Predict the gender
    predict_gender_NB(train_df, test_df)

    # Predict the age
    predict_age_NB(train_df, test_df)
   
    # Predict Personality
    predict_open_LR(train_df, test_df)
    predict_conscientious_LR(train_df, test_df)
    predict_extrovert_LR(train_df, test_df)
    predict_agreeable_LR(train_df, test_df)
    predict_neurotic_LR(train_df, test_df)

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

    # print(f"All the XML files for each userId have been created in the '{output_dir}' directory.")

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