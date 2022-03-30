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
  x = df['Name'].str.rsplit(",", n = 1).str[-1]
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


def feature_engineering(df, meanAge):
  df['Title'] = creation_variable_titre(df)
  df['Title'] = df['Title'].replace('Dona.', 'Mrs.')
  #affichage des valeurs distinctes obtenues pour le 1er mot après la , dans les 2 dataset
  print(df['Title'].unique())
  df['Age'] = df['Age'].fillna(meanAge)
  df['Ticket_Len'] = df['Ticket'].str.len()
  # On s'y connait pas plus sur fare mais on doit la traiter car le dataset de test a une valeur null même sort que l'age on lui met une moyenne
  df['Fare'] = df['Fare'].fillna(df['Fare'].mean())
  # Le nombre de valeur null étant importante on va ajouter la variable hasCabin 1 ou 0 pour ne retenir que si la personne avait une cabine ou non, la encore en se renseignant peut etre que la numérotation des cabines avaient un sens plus précis.
  df['hasCabin'] = df['Cabin'].notnull().astype(int)
  # il a 2 null value dans Embarked qu'on ajoute à la valeur la plus fréquente S
  df['Embarked'] = df['Embarked'].fillna('S')
  df.drop(
    ['PassengerId', 'Name', 'Ticket', 'Cabin'],
    axis = 1, inplace = True)
  return df


def import_clean_data():
    TrainingData = pd.read_csv('train.csv')
    TestData = pd.read_csv('test.csv')
    passengerId = TestData['PassengerId']
    meanAge=round(TrainingData['Age'].mean())
    TrainingData = feature_engineering(TrainingData, meanAge)
    TestData = feature_engineering(TestData, meanAge)
    return {"train": TrainingData, "test": TestData}