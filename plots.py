import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import f_classif

nltk.download('stopwords')


def plot_top_tokens(
    df,
    text_column='text',
    top_n=20,
    remove_stopwords=True
):
    # стоп-слова
    stop_words = set(stopwords.words('english')) if remove_stopwords else set()

    # токенизация
    def tokenize(text):
        text = text.lower()
        tokens = re.findall(r'\b[a-z]+\b', text)
        if remove_stopwords:
            tokens = [t for t in tokens if t not in stop_words]
        return tokens

    # применяем
    tokens_series = df[text_column].apply(tokenize)

    # "сплющиваем"
    all_tokens = [t for tokens in tokens_series for t in tokens]

    # считаем частоты
    counts = Counter(all_tokens)
    top_tokens = counts.most_common(top_n)

    # в DataFrame для удобства
    tokens_df = pd.DataFrame(top_tokens, columns=['token', 'count'])

    # plot
    plt.figure(figsize=(10,12))
    plt.barh(tokens_df['token'], tokens_df['count'])
    
    plt.xlabel("Frequency")
    plt.ylabel("Tokens")
    plt.title(f"Top {top_n} tokens" + (" (no stopwords)" if remove_stopwords else ""))
    
    plt.gca().invert_yaxis()  # чтобы самые частые были сверху
    plt.tight_layout(pad=3)
    plt.show()


def plot_top_features_anova_signed(
    df,
    text_column='text',
    label_column='label',
    top_n=20,
    ngram_range=(1, 2),
    max_features=5000,
    remove_stopwords=True
):
    """
    Строит топ_n самых информативных слов/фраз с помощью ANOVA F-test,
    с учётом направления (positive vs negative).
    
    Параметры:
    - df: DataFrame с текстом и метками
    - text_column: название колонки с текстом
    - label_column: название колонки с метками
    - top_n: сколько топовых признаков показать
    - ngram_range: диапазон n-грамм
    - max_features: максимальное количество признаков
    - remove_stopwords: удалять ли стоп-слова
    """
    
    # TF-IDF
    vectorizer = TfidfVectorizer(
        stop_words='english' if remove_stopwords else None,
        ngram_range=ngram_range,
        max_features=max_features
    )
    
    X = vectorizer.fit_transform(df[text_column])
    y_array = df[label_column].values
    
    feature_names = vectorizer.get_feature_names_out()
    
    # ANOVA F-test
    F_scores, _ = f_classif(X, y_array)
    
    # направление (среднее по классам)
    pos_mean = X[y_array == 1].mean(axis=0).A1
    neg_mean = X[y_array == 0].mean(axis=0).A1
    direction = pos_mean - neg_mean
    
    # объединяем
    features_df = pd.DataFrame({
        'feature': feature_names,
        'F_score': F_scores,
        'direction': direction
    })
    
    # signed importance
    features_df['signed_score'] = features_df['F_score'] * features_df['direction']
    
    # топ по модулю
    top_features = features_df.reindex(
        features_df['signed_score'].abs().sort_values(ascending=False).index
    ).head(top_n)
    
    # сортировка для графика
    top_features = top_features.sort_values('signed_score')
    
    # Plot
    plt.figure(figsize=(10, top_n * 0.4))
    plt.barh(top_features['feature'], top_features['signed_score'])
    plt.xlabel("Signed importance (negative ← → positive)")
    plt.ylabel("Tokens / n-grams")
    plt.title(f"Top {top_n} features (ANOVA + direction)")
    plt.tight_layout(pad=2)
    plt.show()


def heat_map(df):

    plt.figure(figsize=(10, 4))
    
    plt.imshow(df, aspect='auto')
    
    plt.xticks(range(len(df.columns)), df.columns)
    plt.yticks(range(len(df.index)), df.index)
    
    # подписи значений
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            plt.text(j, i, f"{df.iloc[i, j]:.3f}",
                     ha='center', va='center')
    
    plt.title("Model Comparison (5-fold CV)")
    plt.colorbar()
    plt.tight_layout()
    plt.show()

def plot_top_features(pipe, model_name, top_n=20, show_sign=True):
    """
    pipe: обученный pipeline
    model_name: "LogReg", "RandomForest", "XGBoost"
    top_n: сколько топ признаков показать
    show_sign: если True, учитывать знак для LogisticRegression
    """
    
    # Получаем feature names из CountVectorizer/TfidfVectorizer
    vectorizer = pipe.named_steps['preprocessor'].named_transformers_['text']
    feature_names = vectorizer.get_feature_names_out()
    
    if model_name == "LogReg":
        coefs = pipe.named_steps['model'].coef_[0]
        if show_sign:
            importance = coefs
        else:
            importance = np.abs(coefs)
    
    elif model_name == "RandomForest":
        # преобразуем sparse → dense, если есть
        if hasattr(pipe.named_steps['model'], 'feature_importances_'):
            importance = pipe.named_steps['model'].feature_importances_
        else:
            raise ValueError("RandomForest не обучен или нет feature_importances_")
    
    elif model_name == "XGBoost":
        booster = pipe.named_steps['model'].get_booster()
        
        # Получаем важности по "gain"
        importance_dict = booster.get_score(importance_type='gain')  # словарь f0, f1...
        
        # Сопоставляем по порядку признаков
        #num_features = pipe.named_steps['preprocessor'].transform(X_train).shape[1]
        num_features = pipe.named_steps['model'].n_features_in_
        importance = np.zeros(num_features)
        
        for f_name, val in importance_dict.items():
            # f_name = "f123" → индекс
            idx = int(f_name[1:])
            importance[idx] = val
    
    else:
        raise ValueError("Модель должна быть 'LogReg', 'RandomForest' или 'XGBoost'")
    
    # Создаём DataFrame
    df_feat = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    })
    
    # сортировка по абсолютной важности
    df_feat['abs_imp'] = np.abs(df_feat['importance'])
    df_feat = df_feat.sort_values(by='abs_imp', ascending=False).head(top_n)
    
    # Барплот
    plt.figure(figsize=(8, top_n*0.4))
    plt.barh(df_feat['feature'], df_feat['importance'], color='skyblue')
    plt.xlabel('Importance' + (' (signed)' if show_sign and model_name=="LogReg" else ''))
    plt.ylabel('Feature')
    plt.title(f"Top {top_n} features: {model_name}")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()



