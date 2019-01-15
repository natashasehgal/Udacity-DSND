import sys
from sqlalchemy import create_engine

import re
import numpy as np
import pandas as pd
import string

from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.externals import joblib
import pickle

import nltk
from nltk.corpus import wordnet
from nltk import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download('punkt')
nltk.download('wordnet')

ind = 0
def load_data(database_filepath):
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('PipelineTable',engine)
    global ind 
    ind = df['index']
      
    X = df['message']
    Y = df.drop(['index','id','message','original','genre'],axis=1)
    categories = Y.columns.tolist()
    print(categories)
    return X,Y,categories

def tokenize(text):
    text =  " ".join("".join([" " if ch in string.punctuation else ch for ch in text]).split())
    clean_tokens = []
    punctuations = list(string.punctuation)
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer = tokenize)),
                         ('tfidf', TfidfTransformer()),
                        ('clf',MultiOutputClassifier(RandomForestClassifier()))])
    
       
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.5,0.75,1.0),
        'vect__max_features': (None, 5000),
        'tfidf__use_idf': (True, False)
    }
    cv = GridSearchCV(pipeline, param_grid=parameters,n_jobs=4,cv = 3,verbose=10)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    Y_test = Y_test.values
    for i,cols in enumerate(category_names):
        print('COLUMN: ', cols.upper())
        y_test = Y_test.values[:,i]
        y_pred = Y_pred[:,i]
        print('Accuracy: ',accuracy_score(y_test,y_pred))
        res = precision_recall_fscore_support(y_test,y_pred,average='macro',labels=[0,1]) 
        print('Precision: ',res[0])
        print('Recall: ', res[1])
        print('Fscore: ', res[2])


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, "wb"))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        
        
    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()