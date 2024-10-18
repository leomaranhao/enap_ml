import optuna

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

from imblearn.under_sampling import RandomUnderSampler


# FYI: Objective functions can take additional arguments
# (https://optuna.readthedocs.io/en/stable/faq.html#objective-func-additional-args).
def objective(trial):
    
    # Carregar Dados
    df = pd.read_csv('Adult_census_income.csv', na_values = '?')

    # Remover dados nulos
    df = df.dropna()

    df['income'] = df['income'].replace({'<=50K.': '<=50K', '>50K.': '>50K'})

    # Identificação de variáveis categóricas e numéricas
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Remover a variável target e educação original
    categorical_features.remove('income')

    # Mapeamento para agrupar education
    education_mapping = {
        'Preschool': 'Elementary',
        '1st-4th': 'Elementary',
        '5th-6th': 'Elementary',
        '7th-8th': 'Elementary',
        '9th': 'Less than High School',
        '10th': 'Less than High School',
        '11th': 'Less than High School',
        '12th': 'Less than High School',
        'HS-grad': 'High School',
        'Some-college': 'Some College',
        'Assoc-voc': 'Associate Degree',
        'Assoc-acdm': 'Associate Degree',
        'Bachelors': 'Bachelors',
        'Masters': 'Masters',
        'Prof-school': 'Professional School',
        'Doctorate': 'Doctorate'
    }

    # Aplicar o mapeamento na coluna education e adicionar nova coluna 'education_grouped' com os valores
    df['education'] = df['education'].replace(education_mapping)

    # Definir a pipeline de pré-processamento

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='passthrough'
    )

    # Dividir os dados em treino e teste
    X = df.drop('income', axis=1)
    y = df['income'].apply(lambda x: 1 if x == '>50K' else 0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Treinar o modelo com HyperOpt
    undersampler = RandomUnderSampler(random_state=42)

    X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train, y_train)

    X_train_resampled = preprocessor.fit_transform(X_train_resampled)
    X_test = preprocessor.transform(X_test)

    X_train_resampled = pd.DataFrame(X_train_resampled, columns=preprocessor.get_feature_names_out())

    X_test = pd.DataFrame(X_test, columns=preprocessor.get_feature_names_out())

    classifier_name = trial.suggest_categorical("classifier", ["XGBoost", "LogisticRegression", "RandomForest"])
    if classifier_name == "LogisticRegression":
        lr_warm_start =  trial.suggest_categorical("lr_warm_start", [True, False])
        lr_fit_intercept =  trial.suggest_categorical("lr_fit_intercept", [True, False])
        lr_tol = trial.suggest_float("lr_tol", 0.00001, 0.0001, log=True)
        lr_C = trial.suggest_float("lr_C", 0.05, 3, log=True)
        lr_solver = trial.suggest_categorical("lr_solver", ['newton-cg', 'lbfgs', 'liblinear'])
        lr_max_iter = trial.suggest_int("lr_max_iter", 50, 1000)
            
        classifier_obj = LogisticRegression(
            C=lr_C, class_weight='balanced', random_state=42, warm_start=lr_warm_start, 
            fit_intercept=lr_fit_intercept, tol=lr_tol,solver=lr_solver, max_iter=lr_max_iter)
        
    elif classifier_name == "XGBoost":
        xg_n_estimators =  trial.suggest_int("xg_n_estimators", 100, 400, step=50)
        xg_max_depth = trial.suggest_int("xg_max_depth", 2, 10)
            
        classifier_obj = XGBClassifier(
            random_state=42, eval_metric='logloss', max_depth=xg_max_depth, n_estimators=xg_n_estimators, 
            learning_rate=0.1)
        
    else:
        rf_max_depth = trial.suggest_int("rf_max_depth", 2, 32)
        rf_max_features = trial.suggest_int("rf_max_features", 1, 3)
        rf_n_estimators = trial.suggest_int("rf_n_estimators", 10, 50)
        rf_criterion = trial.suggest_categorical("rf_criterion", ["gini", "entropy"])

        classifier_obj = RandomForestClassifier(
            max_depth=rf_max_depth, n_estimators=rf_n_estimators, max_features=rf_max_features, 
            criterion=rf_criterion, random_state=42
        )

    score = cross_val_score(classifier_obj, X_train_resampled, y_train_resampled, n_jobs=-1, cv=3, scoring='f1')
    accuracy = score.mean()

    return accuracy


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)
    print('\n')
    print(study.best_trial)