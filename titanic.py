import pandas as pd ; import numpy as np
import matplotlib.pyplot as plt ; import seaborn as sns
import multiprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import pathlib



from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import time


import os
print(os.getcwd())
os.chdir("ensae-reproductibilite-projet/")

TrainingData = pd.read_csv('train.csv')
TestData = pd.read_csv('test.csv')

# PARTIE 2 ------------------------------------------------
# Un peu d'exploration et de feature engineering

# Recodage de certaines variables ========================

# Variable identifiant : on la stocke pour le dataset de test mais on la supprime du dataset l'identifiant n'ayant un sens que pour nous retrouver l'indvidu. Kaggle proposant d'uploader notre prédiction si l'on souhaite pour donner le score de notre algorithme.
passengerId = TestData['PassengerId']
TrainingData.drop(labels='PassengerId', axis=1, inplace=True)
TestData.drop(labels='PassengerId', axis=1, inplace=True)

TrainingData.columns

fig, axes=plt.subplots(1,2, figsize=(12, 6)) #layout matplotlib 1 ligne 2 colonnes taile 16*8
fig1_pclass=sns.countplot(data=TrainingData, x ="Pclass",    ax=axes[0]).set_title("fréquence des Pclass")
fig2_pclass=sns.barplot(data=TrainingData, x= "Pclass",y= "Survived", ax=axes[1]).set_title("survie des Pclass")

# Extraction et ajout de la variable titre
def creation_variable_titre(df: pd.DataFrame, var: str = "Name"):
  x = TrainingData['Name'].str.rsplit(",", n = 1).str[-1]
  x = x.str.split().str[0]
  #On note que Dona est présent dans le jeu de test à prédire mais dans les variables d'apprentissage on règle ca a la mano
  return x

TrainingData['Title'] = creation_variable_titre(TrainingData)
TestData['Title'] = creation_variable_titre(TestData)

#affichage des valeurs distinctes obtenues pour le 1er mot après la , dans les 2 dataset
print(TrainingData['Title'].unique())
print(TestData['Title'].unique())

TestData['Title'] = TestData['Title'].replace('Dona.', 'Mrs.')


# Suppression de la variable Titre
TrainingData.drop(labels='Name', axis=1, inplace=True)
TestData.drop(labels='Name', axis=1, inplace=True)



fx, axes = plt.subplots(2, 1, figsize=(15, 10))
fig1_title = sns.countplot(data=TrainingData, x='Title', ax=axes[0]).set_title("Fréquence des titres")
fig2_title = sns.barplot(data=TrainingData, x='Title',y='Survived', ax=axes[1]).set_title("Taux de survie des titres")

# On va pas se faire suer pour notre part on va mettre la moyenne de l'age sur le bateau quand on ne la connait pas.
sns.distplot(a= TrainingData['Age'].dropna(axis = 0),bins = 15,hist_kws={'rwidth'     :0.7}).set_title("distribution de l'age")


meanAge=round(TrainingData['Age'].mean())
TrainingData['Age'] = TrainingData['Age'].fillna(meanAge)
TestData['Age'] = TrainingData['Age'].fillna(meanAge)

# Making a new feature ticket length
TrainingData['Ticket_Len'] = TrainingData['Ticket'].apply(lambda x: len(x))
TestData['Ticket_Len'] = TestData['Ticket'].apply(lambda x: len(x))
TrainingData.drop(labels='Ticket', axis=1, inplace=True)
TestData.drop(labels='Ticket', axis=1, inplace=True)

# On s'y connait pas plus sur fare mais on doit la traiter car le dataset de test a une valeur null même sort que l'age on lui met une moyenne
TestData['Fare']=TestData['Fare'].fillna(TestData['Fare'].mean())

# Le nombre de valeur null étant importante on va ajouter la variable hasCabin 1 ou 0 pour ne retenir que si la personne avait une cabine ou non, la encore en se renseignant peut etre que la numérotation des cabines avaient un sens plus précis.
TrainingData['hasCabin'] = TrainingData.Cabin.notnull().astype(int)
TestData['hasCabin'] = TestData.Cabin.notnull().astype(int)

TrainingData.drop(labels='Cabin', axis=1, inplace=True)
TestData.drop(labels='Cabin', axis=1, inplace=True)

TrainingData.columns, TestData.columns


# il a 2 null value dans Embarked qu'on ajoute à la valeur la plus fréquente S
TrainingData['Embarked'] = TrainingData['Embarked'].fillna('S')
TestData['Embarked'] = TestData['Embarked'].fillna('S')

#A ce stade on est "bon" sur le contenu des variables, il n'y a plus de valeurs null dans aucun des dataset
print(    TrainingData.isnull().sum())
print(TestData.isnull().sum()   )


# PARTIE 2: Encoder les données imputées ou transformées. ---------------------------

label_encoder_sex = LabelEncoder()
label_encoder_title = LabelEncoder()
label_encoder_embarked = LabelEncoder()
TrainingData['Sex'] = label_encoder_sex.fit_transform(TrainingData['Sex'].values)
TrainingData['Title'] = label_encoder_title.fit_transform(TrainingData['Sex'].values)
TrainingData['Embarked'] = label_encoder_embarked.fit_transform(TrainingData['Sex'].values)

TrainingData.head()

#On va maintenant passer du monde panda au monde numpy pour servir d'input à l'apprentissage pour cela on isole la variable cible
y = TrainingData.iloc[:, 0].values
#et le reste du dataset
X = TrainingData.iloc[:, 1:12].values

# Feature Scaling
scaler_x = MinMaxScaler((-1,1))
X = scaler_x.fit_transform(X)

# Prenons arbitrairement 10% du dataset en test et 90% pour l'apprentissage
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# PARTIE 3: Random Forest ------------------------------------

## 3.1. Initialisation =====================

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import pathlib
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

#Ici demandons d'avoir 20 arbres
rdmf = RandomForestClassifier(n_estimators=20)
rdmf.fit(X_train, y_train)

#calculons le score sur le dataset d'apprentissage et sur le dataset de test (10% du dataset d'apprentissage mis de côté)
# le score étant le nombre de bonne prédiction
rdmf_score = rdmf.score(X_test, y_test)
rdmf_score_tr = rdmf.score(X_train, y_train)
print("{} % de bonnes réponses sur les données de test pour validation (résultat qu'on attendrait si on soumettait notre prédiction sur le dataset de test.csv)".format(round(rdmf_score*100)))
from sklearn.metrics import confusion_matrix
print("matrice de confusion")
confusion_matrix(y_test, rdmf.predict(X_test))

## 3.2. HyperParamètres =====================

from sklearn.model_selection import GridSearchCV

rf = RandomForestClassifier(max_features='auto')

param_grid = {
       "criterion" : ["gini", "entropy"], 
    "min_samples_leaf" : [1, 5, 10], 
      "min_samples_split" : [2, 4, 10, 12], 
    "n_estimators": [50, 100, 400, 700]}
#par défaut il va appliquer une default 5-fold cross validation,
gs = GridSearchCV(estimator=rf, param_grid=param_grid, scoring='accuracy', cv=3, n_jobs=-1)

gs = gs.fit(X, y)

print(" {} % score attendue".format(gs.best_score_))
print("paramètres retenus", gs.best_params_)

print(gs.best_estimator_.feature_importances_)
print(TrainingData.columns[1:])
import pandas as pd

#do code to support model
#"data" is the X dataframe and model is the SKlearn object
feats = {} # a dict to hold feature_name: feature_importance
for feature, importance in zip(TrainingData.columns[1:], gs.best_estimator_.feature_importances_):
      feats[feature] = importance #add the name/value pair 

importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'importance'})
importances.sort_values(by='importance').plot(kind='bar', rot=45)
