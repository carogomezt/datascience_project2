import sys

import nltk

nltk.download(['punkt', 'wordnet', 'stopwords'])

import re
import pandas as pd
from sqlalchemy import create_engine

import pickle

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def load_data(database_filepath):
    """
    Load data from the sqlite DataBase.
    :param database_filepath: String with the route of the database.
    :return X, Y:  DataFrames with the data.
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql("SELECT * FROM messages", engine)
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    return X, Y


def tokenize(text):
    """
    Tokenize the text input removing stopwords and lemmatizing the text.
    :param text: String with the text data.
    :return: List with all the tokens.
    """
    # normalize case and remove punctuation
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text.lower())
    # tokenize text
    tokens = word_tokenize(text)

    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    # lemmatize and remove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return tokens


def build_model(optimize=False):
    """
    Generates a model.
    :param optimize: Boolean that enables the GridSearch optimization.
    :return: Pipeline with the model.
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    if optimize:
        parameters = {
            'clf__estimator__n_estimators': [100, 200, 300],
            'clf__estimator__min_samples_split': [2, 5, 10],
            'clf__estimator__max_features': ['auto', 'sqrt'],
            'clf__estimator__bootstrap': [True, False],
        }

        pipeline = GridSearchCV(pipeline, param_grid=parameters, cv=3, verbose=3)

    return pipeline


def evaluate_model(model, X_test, Y_test):
    """
    Shows different metrics to evaluate the model performance
    :param model: Pipeline with the model
    :param X_test: DataFrame with the X values to test.
    :param Y_test: DataFrame with the Y values to test.
    :return: None
    """
    y_pred = model.predict(X_test)
    for ind, cat in enumerate(Y_test):
        print(f'Target Category - {cat}')
        print(classification_report(Y_test.values[ind], y_pred[ind], zero_division=1))

    # Model score
    model_score = model.score(X_test, Y_test)
    print(f'Model Score: {model_score}')


def save_model(model, model_filepath):
    """
    Save model in the given file path.
    :param model: Pipeline with the model.
    :param model_filepath: String the the route to save the model.
    :return:
    """
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model(optimize=False)
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()