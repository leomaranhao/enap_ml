# Importar bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from hyperopt import hp, tpe, fmin, Trials, STATUS_OK
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import scale, normalize
from sklearn.model_selection import cross_val_score

from hpsklearn import HyperoptEstimator, xgboost_classification, random_forest_classifier, logistic_regression

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

from imblearn.under_sampling import RandomUnderSampler

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

clf = hp.pchoice( 'my_name', 
          [ ( random_forest_classifier('my_name.random_forest_classifier', random_state=42, class_weight='balanced') ),
            ( xgboost_classification('my_name.xgboost', random_state=42) ),
            ( logistic_regression('my_name.logistic_regression', random_state=42) ) ])
                  
estim = HyperoptEstimator( classifier=clf, 
                            preprocessing=[],
                            algo=tpe.suggest, trial_timeout=300)

estim.fit( X_train_resampled, y_train_resampled )
estim.score(X_test, y_test)