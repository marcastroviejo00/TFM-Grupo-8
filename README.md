# Trabajo Final de Máster Grupo 8

Este repositorio contiene el código y análisis realizados como parte del Trabajo Final de Máster (TFM). 

El Trabajo de Fin de Máster (TFM) se centra en la modelización del siguiente dataset disponible en Kaggle, cuyo objetivo principal es realizar un benchmarking de diferentes modelos predictivos.
https://www.kaggle.com/datasets/ara001/bank-client-attributes-and-marketing-outcomes/data

## Descripción del Proyecto

El proyecto se divide en varias secciones clave:

1. **Exploración de Datos**: Se lleva a cabo un análisis exploratorio de los datos, donde se examinan las características principales del dataset.
2. **Tratamiento de Datos**: Se incluyen técnicas de preprocesamiento como el manejo de valores nulos, escalado y transformación de características.
3. **Modelización**: Aplicación de diversos modelos de machine learning. Decision Tree, Naive Bayes, KNN, Logistic Regression, SVM y la red neuronal MLP.
4. **Análisis de Modelos**: Evaluación y comparación de los resultados obtenidos por los diferentes modelos con cross validation y la curva ROC.

## Requisitos Previos

Para ejecutar correctamente el código, es necesario tener instaladas las siguientes dependencias e importar las librerías correspondientes (se encuentran en la primera y segunda celda del código):

!pip install scikit-learn==1.1.2 scipy==1.9.1 missingpy==0.2.0

--------------------------------------------------

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier

import sklearn.neighbors._base

import sys

sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base

from missingpy import MissForest

from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, ConfusionMatrixDisplay, accuracy_score, make_scorer, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc

from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, KFold, RandomizedSearchCV

from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, label_binarize

from sklearn.compose import ColumnTransformer

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
