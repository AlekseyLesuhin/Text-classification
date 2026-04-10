import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def build_model(model_type, vocab_size, max_len, lr, embedding_matrix=None):
    """
    model_type: 'rnn', 'lstm' или 'cnn'
    """
    
    model = Sequential()

    if embedding_matrix is not None:
        model.add(Embedding(input_dim=embedding_matrix.shape[0],
                            output_dim=embedding_matrix.shape[1],
                            input_length=max_len,
                            weights=[embedding_matrix],
                            trainable=False))
    else:
        model.add(Embedding(input_dim=vocab_size, output_dim=100, input_length=max_len))
    
    # Выбор архитектуры
    if model_type == 'rnn':
        model.add(SimpleRNN(64))
        
    elif model_type == 'lstm':
        model.add(LSTM(64, return_sequences=True))
        model.add(GlobalMaxPooling1D())

    elif model_type == 'cnn':
        model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
        model.add(GlobalMaxPooling1D())
        
    else:
        raise ValueError("model_type должен быть 'rnn', 'lstm' или 'cnn'")
    
    # Общая часть
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=lr),
        metrics=['accuracy']
    )
    
    return model


def train_model_cv(type_of_model, X_train, y_train, MAX_WORDS, MAX_LEN, learn_rate, embedding_matrix=None):
    lr = learn_rate
    model_type = type_of_model
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    acc_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    roc_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"Fold {fold+1}")
        
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        model = build_model(model_type, MAX_WORDS, MAX_LEN, lr, embedding_matrix = embedding_matrix)
        
        model.fit(
            X_tr, y_tr,
            epochs=30,
            batch_size=32,
            validation_data=(X_val, y_val),
            verbose=1
        )
        
        # предсказания
        y_pred_proba = model.predict(X_val).ravel()
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # метрики
        acc_scores.append(accuracy_score(y_val, y_pred))
        precision_scores.append(precision_score(y_val, y_pred))
        recall_scores.append(recall_score(y_val, y_pred))
        f1_scores.append(f1_score(y_val, y_pred))
        roc_scores.append(roc_auc_score(y_val, y_pred_proba))

    print(f"CV Results ({type_of_model}):")
    print(f"Accuracy: {np.mean(acc_scores):.4f}")
    print(f"Precision: {np.mean(precision_scores):.4f}")
    print(f"Recall: {np.mean(recall_scores):.4f}")
    print(f"F1-score: {np.mean(f1_scores):.4f}")
    print(f"ROC-AUC: {np.mean(roc_scores):.4f}")

    return [np.mean(acc_scores), np.mean(precision_scores), np.mean(recall_scores), np.mean(f1_scores), np.mean(roc_scores)]


def load_glove_embeddings(glove_path, embedding_dim, tokenizer, MAX_WORDS):
    embeddings_index = {}
    with open(glove_path, encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    
    embedding_matrix = np.zeros((MAX_WORDS, embedding_dim))
    for word, i in tokenizer.word_index.items():
        if i >= MAX_WORDS:
            continue
        vector = embeddings_index.get(word)
        if vector is not None:
            embedding_matrix[i] = vector
    return embedding_matrix

