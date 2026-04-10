import pandas as pd
import numpy as np
import string
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import cross_validate
nltk.download('wordnet')

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def evaluate_models_cv(models, X_train, y_train, X_test, y_test, cv):
    
    results = {}

    scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'roc_auc': 'roc_auc'}
    
    for name, model in models.items():
        cv_results = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )
        
        results[name] = {
            metric: np.mean(cv_results[f'test_{metric}'])
            for metric in scoring.keys()
        }

    print("Test Results:")

    for name, model in models.items():
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        print(f"{name}: {score:.4f}")
    
    return pd.DataFrame(results).T