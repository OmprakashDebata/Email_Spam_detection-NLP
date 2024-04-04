# Email Spam detection using NLP

This project aims to detect email spam using Natural Language Processing (NLP) techniques. The classification model is built using Support Vector Machine (SVM) algorithm on the basis of TF-IDF vectorization of email messages.

## Table of Contents:

1 - Introduction

2 - Project Structure

3 - Data Preprocessing

4 - Exploratory Data Analysis (EDA)

5 - Model Building

6 - Evaluation

7 - Deployment

8 - Usage

9 - Conclusion

## Introduction

Spam emails are a common nuisance and can sometimes be harmful. Detecting spam emails automatically can save time and reduce security risks. This project uses NLP techniques to preprocess email messages, extract relevant features, and build a classification model to distinguish between spam and non-spam emails.

## Project Structure :

- email_spam_detection.ipynb: Jupyter Notebook containing the code for data preprocessing, model building, and evaluation.
- Email_spam_detect.pkl: Pickle file containing the trained SVM model.
- README.md: This file providing an overview of the project.
- messages.csv: Dataset containing email messages labeled as spam or non-spam.

## Data Preprocessing :

The dataset containing email messages is preprocessed to clean and standardize the text data. This includes removing special characters, URLs, email addresses, and applying lowercase conversion. Additionally, stopwords are removed, and text is tokenized for further analysis.

## Exploratory Data Analysis (EDA) :

Exploratory Data Analysis is performed to gain insights into the distribution of spam and non-spam messages, message length distributions, and word clouds for both spam and non-spam messages.

## Model Building :

The classification model is built using the Support Vector Machine (SVM) algorithm. The TF-IDF vectorization technique is applied to convert text data into numerical features. The SVM model is trained on these features to classify emails as spam or non-spam.

## Evaluation :

The performance of the SVM model is evaluated using metrics such as accuracy, precision, recall, and F1-score. Additionally, a confusion matrix is generated to visualize the model's performance on test data.

## Deployment :
 
The trained SVM model is serialized using pickle and saved as Email_spam_detect.pkl for future use or deployment in production environments.

## Usage :

To use the trained model for email spam detection:
Load the serialized SVM model (Email_spam_detect.pkl).
Preprocess the email messages using the same preprocessing steps as in the notebook.
Vectorize the preprocessed text data using TF-IDF vectorization.
Predict the labels (spam or non-spam) using the trained SVM model.

## Conclusion :

Email spam detection using NLP techniques and machine learning algorithms such as SVM can be effective in identifying and filtering out unwanted and potentially harmful emails. By preprocessing text data, extracting relevant features, and training a classification model, we can achieve reliable spam detection results.

In this project, we successfully built and trained an SVM model on email message data. The model demonstrated good performance in distinguishing between spam and non-spam emails, as evidenced by the evaluation metrics such as accuracy, precision, recall, and F1-score.

By deploying the trained model, organizations and individuals can enhance their email security measures, reduce the risk of falling victim to phishing attacks, and improve overall productivity by avoiding unnecessary distractions from spam emails.


If you have any feedback, please reach out to me at omprakashdebata12@gmail.com


