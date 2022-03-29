import pandas as pd ; import numpy as np
import matplotlib.pyplot as plt ; import seaborn as sns
import multiprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import pathlib



from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import time


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import pathlib
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV



# Extraction et ajout de la variable titre
def creation_variable_titre(df: pd.DataFrame, var: str = "Name"):
  x = TrainingData['Name'].str.rsplit(",", n = 1).str[-1]
  x = x.str.split().str[0]
  #On note que Dona est présent dans le jeu de test à prédire mais dans les variables d'apprentissage on règle ca a la mano
  return x

def create_figure_frequence(df: pd.DataFrame, xvar: str):
    fig, axes=plt.subplots(1,2, figsize=(12, 6)) #layout matplotlib 1 ligne 2 colonnes taile 16*8
    fig1_pclass=sns.countplot(data=df, x = xvar,    ax=axes[0]).set_title(f"fréquence des {xvar}")
    fig2_pclass=sns.barplot(data=df, x= xvar, y= "Survived", ax=axes[1]).set_title(f"survie des {xvar}")

def label_encode_variable(df: pd.DataFrame, var: str = "Sex"):
  encoder = LabelEncoder()
  df[var] = encoder.fit_transform(df[var].values)
  return df