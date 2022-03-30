# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + [markdown] _cell_guid="88a560b0-7a9d-448c-8ef2-deacb6fb37f1" _uuid="a85f44497092d3c8a51bd913880f626d59f57e4c"
# # Prediction de la survie d'un individu sur le Titanic
#
# Ce tutoriel repose sur les données et le défi exemple de la communauté kaggle sur les données du titanic.
#
# Il s'agit à partir de la liste des passagers du titanic et de leur survie ou non de prédire la chance de survie d'un individu en fonction de son nom, age, sexe, situation familiale, économique...
#
# Ce notebook est inspiré par https://www.kaggle.com/mukultiwari/titanic-top-14-with-random-forest
#

# + _cell_guid="b11366bd-b985-4df1-9630-b2d57f60d0f0" _uuid="2da21fe9ea560ec76cded00d85f84caa7932a126"
import pandas as pd ; import numpy as np
import matplotlib.pyplot as plt ; import seaborn as sns
import multiprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import pathlib



from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import time


# %matplotlib inline
#import warnings
#warnings.filterwarnings('ignore')

# + [markdown] _cell_guid="e63ba51a-a82b-4e75-9f73-6ff5374b49f8" _uuid="8063f112062a792a052c7b8c3b07ec32a9ba201f"
# # Données
#
# On va utiliser la librairie pandas pour lire les données, on y a déposé le jeu de donnée disponible sur kaggle :
# * le train.csv contient les données d'apprentissages, c'est à dire les variables décrivants des individus et s'ils ont survécus ou non.
# * le test.csv contient uniquement la description d'individu dont il faut prédire leur chance de survie. 

# + [markdown] _cell_guid="dd58ece3-bf62-484b-a7b8-9671a874c7e3" _uuid="56224632f5ef18fe17364b808faaa1f5725f4079"
# ---

# + _cell_guid="096ddef0-fc1a-4a65-bd56-82e5b3ef3d96" _uuid="fdcb49f7845aae42bceb3216b3a2f09374c762ba"
import os

TrainingData = pd.read_csv('train.csv')
TestData = pd.read_csv('test.csv')

# + _cell_guid="b02f3b71-dd0f-4c22-8e48-fc4ee8616e21" _uuid="5ceeb0b9a9ddd42db685a5e5f0503e01788a016a"
TrainingData.head()

# + _cell_guid="afb69ad3-59ea-4fe5-bd96-05e33638e9bd" _uuid="c75a5e514a13817d8d2ab2c70f6e902f4a27669f"
TrainingData.info()
TestData.info()

# + [markdown] _cell_guid="68fb208f-6c43-47c9-8716-077caef83898" _uuid="ab2c8c5382d158f105ea97bff118f22022d236ad" jupyter={"outputs_hidden": true}
# On a :
# * 7 variables numériques : **PassengerId, Survived, Pclass, Age, SibSp, Parch, Fare**
#     * dont **PassengerId** qui est l'identifiant du passager dans le dataset
#     * dont **Survived** qui est la variable à prédire la variable "cible"
#     * dont PClass qui est une indication de la strate socio économique
#     * dont  SibSp et Parch qui permettent de déterminer la situation familiale(epoux, mère, fille...)
#     * Fare qui est un prix ou quelque chose comme ca, traitons la sans la comprendre c'est pas la première et dernière fois que ca arrivera.
# * 5 variables non numériques : **Name, Sex, Ticket** un identifiant de ticket, **Cabin** un identifiant de cabin, **Embarked** le port d'embarquation C = Cherbourg, Q = Queenstown, S = Southampton)
#
# Recherchons s'il y a des valeurs manquantes dans ces 2 dataset

# + _cell_guid="f6bd235e-5670-4d21-b0eb-bececfc1757a" _uuid="6c44f4bd0762061bedd908dcbe854b14ae9f0966"
TrainingData.isnull().sum()

# + _cell_guid="b12b1a12-5563-44ef-90a7-4a77b0a0d3f3" _uuid="7f46a31c1465fb979f9bfada8c17b01e0bd1c850"
TestData.isnull().sum()

# + [markdown] _cell_guid="aaf786a2-4fc4-45e5-a265-b3f7ab422eb4" _uuid="e7ae92c394024a41aaf4a726a20c44ae037544b0"
# Comme généralement en machine learning, il va falloir traiter ces valeurs manquantes en imputant des valeurs :
# * dans le train dataset ( Age 177 manquants, Cabin 687 manquants, Embarked 2 manquants)
# * dans le test dataset ( Age 86 manquants, Cabin 327 manquants, Fare 1 manquant)

# + [markdown] _cell_guid="36660fe0-d8a0-4bb1-83aa-3c6a1b17b830" _uuid="183198b90dca66bcb7850d86996149efee961002"
# ---

# + [markdown] _cell_guid="a7117dc9-e37f-4a9c-9031-7684b1a0bbc4" _uuid="265117f74501dfd481714b1f92f238d0c9f63e6f"
# # Un peu d'exploration et de feature engineering
#
# Avant d'entrainer un modèle, il y a généralement une phase exploratoire du dataset que nous allons réduire ici au minimum.
# Il y a aussi selon le contexte du feature engineering qui est probablement l'une des composantes les plus compliquées du machine learning, il s'agit selon le contexte et le problème de créer des variables qui ont un sens pour contribuer à résoudre notre problématique.
#
# Ici, on peut en faire un exemple naturellement sur le nom de l'invidu et instinctivement en extrayant le titre du nom des individus, mais cela peut etre beaucoup moins naturel selon les problèmes voire même complètement un état de l'art si on prend l'exemple de la modélisation des images pour le machine learning ou il s'agit, par exemple, de proposer des valeurs pondérées par sous division de partie d'image.

# + [markdown] _cell_guid="fe85cb6f-25a4-48d4-91c1-d5d32c507292" _uuid="caf1bd633899fccf71cd6808e24eefb51f337603"
# ---

# + [markdown] _cell_guid="0963b71c-b8a7-442f-88df-f1edfffc7746" _uuid="270510a9041cd8626cf6b1f6d8b8ac5e42b26f39"
# # Passenger Id
#
# Variable identifiant : on la stocke pour le dataset de test mais on la supprime du dataset l'identifiant n'ayant un sens que pour nous retrouver l'indvidu. Kaggle proposant d'uploader notre prédiction si l'on souhaite pour donner le score de notre algorithme.

# + _cell_guid="ad7d0c77-6877-43b4-81fd-fa34dcbc5707" _uuid="524ac419ec5e97933aa8ae960c5bcdf75814a4df"
passengerId = TestData['PassengerId']
TrainingData.drop(labels='PassengerId', axis=1, inplace=True)
TestData.drop(labels='PassengerId', axis=1, inplace=True)

# + _cell_guid="4360ee83-d3e9-478c-bbf6-d15dcc2a2cee" _uuid="5b25df1956d83bca9b94598068996012f74b531d"
TrainingData.columns

# + [markdown] _cell_guid="96cd6010-dee3-4274-8623-55810a509868" _uuid="36e6307241afd092a741aa9243da67128e0737d7"
# # variable : Pclass
#
# Une variable catégorique qui donne une idée de la classe socio-économique de la personne dont on donne un exemple avec seaborn pour visualiser la contribution https://seaborn.pydata.org/
#
# Clairement chaque classe n'avait pas la meme chance de survie, n'est ce pas Jack?

# + _cell_guid="cd1e5c45-6156-4544-b705-b86ff92c6d38" _uuid="e0f993a30dc3620a76fe189334bf418b575f2ae7"
fig, axes=plt.subplots(1,2, figsize=(12, 6)) #layout matplotlib 1 ligne 2 colonnes taile 16*8
fig1_pclass=sns.countplot(data=TrainingData, x ="Pclass",    ax=axes[0]).set_title("fréquence des Pclass")
fig2_pclass=sns.barplot(data=TrainingData, x= "Pclass",y= "Survived", ax=axes[1]).set_title("survie des Pclass")

# + [markdown] _cell_guid="719f39df-37a7-4ee6-8170-c3d1db342251" _uuid="5fa8b2ea908fcffdf4056a288ebaddd7ddde2e80"
# # Name
#
# Nous allons essayer d'extraire du nom le titre qui peut etre présent dans le nom :
# * le titre s'il y e en a du type Mr, Miss, Mrs... identifié comme le 1er mot après la **,**.
#
# Exemple : Heikkinen, **Miss.** Laina	
# -

#affichage des valeurs distinctes obtenues pour le 1er mot après la , dans les 2 dataset
print(TrainingData['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0]).unique())
print(TestData['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0]).unique())

# + _cell_guid="80ac2756-c211-4e2c-96ba-f025afb801d7" _uuid="d72df9859e01eaaab1a3ca2b867b8be42d01b4b3"
# Extraction et ajout de la variable titre
TrainingData['Title'] = TrainingData['Name'].apply( lambda x: x.split(',')[1]).apply(lambda x: x.split()[0])
TestData['Title'] = TestData['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0])
# Suppression de la variable Titre
TrainingData.drop(labels='Name', axis=1, inplace=True)
TestData.drop(labels='Name', axis=1, inplace=True)
#On note que Dona est présent dans le jeu de test à prédire mais dans les variables d'apprentissage on règle ca a la mano
TestData['Title'] = TestData['Title'].replace('Dona.', 'Mrs.')

# + _cell_guid="baa77529-f9e6-476c-8f5a-beb7879429c6" _uuid="96f88953ca5028e7293e610996abb700ecc59c4e"
fx, axes = plt.subplots(2, 1, figsize=(15, 10))
fig1_title = sns.countplot(data=TrainingData, x='Title', ax=axes[0]).set_title("Fréquence des titres")
fig2_title = sns.barplot(data=TrainingData, x='Title',y='Survived', ax=axes[1]).set_title("Taux de survie des titres")

# + [markdown] _cell_guid="b8d8c2d1-06a0-49b4-9fc7-4864fe95a360" _uuid="3258a16e7aee8760f4a4786bec699fb88d34e4d2" jupyter={"outputs_hidden": true}
# L'échelle n'as pas exceptionnelle, mais on voit bien que certains titre comme Mrs,Miss,Master, Lady,Sir était préférable.
#
# Sinon tout le monde a vu le film et sait bien que le capitaine n'a pas survécu..

# + [markdown] _cell_guid="4abb7f15-2666-4157-ac04-e32482ec07eb" _uuid="7ddaedcc65f7a576ff8c393140b162b60bcc9756"
# # Age
#
# il nous faut traiter les valeurs nulls, il y a x facons de les traiter voici pour l'exemple un affichage de la distribution, on pourrait chercher tirer au hasard dans une distribution proche, chercher s'il y a des écarts d'age par exemple à partir du titre Miss pouvant surement etre plus jeune que Lady...
#
# On va pas se faire suer pour notre part on va mettre la moyenne de l'age sur le bateau quand on ne la connait pas.

# + _cell_guid="1477483f-069b-43f3-abff-d9764aa08238" _uuid="4159c5b49bc74aa3792fbd86ef7136498dc4e12d"
sns.distplot(a= TrainingData['Age'].dropna(axis = 0),bins = 15,hist_kws={'rwidth'     :0.7}).set_title("distribution de l'age")
# -

meanAge=round(TrainingData['Age'].mean())
TrainingData['Age'] = TrainingData['Age'].fillna(meanAge)
TestData['Age'] = TrainingData['Age'].fillna(meanAge)

# + [markdown] _cell_guid="3562f170-5f29-4550-b103-9f4280586659" _uuid="855906fea192646de43070da2d359521c2fe3ce1"
# # Ticket
#
# On pourrait regarder le contenu ou se renseigner sur les valeur de l'époque s'ils avaient une signification.
#
# N'étant pas historien, on va mettre betement la longueur de la chaine de caractère comme variable, au pire si elle ne contribue pas on l'excluera ou le modèle l'excluera d'elle même.

# + _cell_guid="79b5313b-ef6e-4492-93c3-0e1cc738c8d7" _uuid="c3ba335f483e65c4868b7c8d0ac13b53127aefbe"
# Making a new feature ticket length

TrainingData['Ticket_Len'] = TrainingData['Ticket'].apply(lambda x: len(x))
TestData['Ticket_Len'] = TestData['Ticket'].apply(lambda x: len(x))
TrainingData.drop(labels='Ticket', axis=1, inplace=True)
TestData.drop(labels='Ticket', axis=1, inplace=True)

# + [markdown] _cell_guid="691fbd94-5afb-4a61-8c6a-c7b00f9b4207" _uuid="62b8790f3670809928c98d40cd20c6812aca0aa6"
# # Fare
#
# On s'y connait pas plus sur fare mais on doit la traiter car le dataset de test a une valeur null même sort que l'age on lui met une moyenne

# + _cell_guid="48f02158-1296-452d-91da-608d5fe57766" _uuid="4bad1ecdea38f903418030df660ab2bcf7ac6933"
TestData['Fare']=TestData['Fare'].fillna(TestData['Fare'].mean())

# + [markdown] _cell_guid="1fcc63f4-120d-44f7-b83b-e557f886cb4c" _uuid="a51ba0ec10346283d668f58c23672fb1d11c3db0"
# # Cabin
#
# Le nombre de valeur null étant importante on va ajouter la variable hasCabin 1 ou 0 pour ne retenir que si la personne avait une cabine ou non, la encore en se renseignant peut etre que la numérotation des cabines avaient un sens plus précis.

# + _cell_guid="0f4bf4d5-5653-45a9-8aff-8363940e2a2a" _uuid="09fea57beddf9b0a3f29c40d89d8097be23046ab"
# Making a new feature hasCabin which is 1 if cabin is available else 0
TrainingData['hasCabin'] = TrainingData.Cabin.notnull().astype(int)
TestData['hasCabin'] = TestData.Cabin.notnull().astype(int)

# + _cell_guid="0f06f4e6-8564-4cc5-b7da-87d5c006e833" _uuid="2cddfeb5cb1aa7e4d44cd1bc53f86480d0c857b5"
TrainingData.drop(labels='Cabin', axis=1, inplace=True)
TestData.drop(labels='Cabin', axis=1, inplace=True)

# + _cell_guid="aa0a773c-be00-42be-bcc8-3c5c3c74b4e4" _uuid="60d5f34f1d67ab680dac6e3458996cb921092e61"
TrainingData.columns, TestData.columns

# + [markdown] _cell_guid="e3f39273-24e1-4f55-b4c5-06d955c3d52f" _uuid="19c71236eaa4e88917aec0f136dff6ac247842c9"
# # Embarked
#
# il a 2 null value dans Embarked qu'on ajoute à la valeur la plus fréquente S

# + _cell_guid="bb967c8a-c320-4a50-a121-d1b743c3bda3" _uuid="d71f618046995492830250c1dc32724d1bd3b9a0"
TrainingData['Embarked'] = TrainingData['Embarked'].fillna('S')
TestData['Embarked'] = TestData['Embarked'].fillna('S')
# -

#A ce stade on est "bon" sur le contenu des variables, il n'y a plus de valeurs null dans aucun des dataset
print(    TrainingData.isnull().sum())
print(TestData.isnull().sum()   )

# + [markdown] _cell_guid="1cefe67c-11d2-4f66-af18-374b3becf563" _uuid="60a2ea05bce4a59c50f2037d68362cbb6d5b5e1a"
# # Encoder les données imputées ou transformées.

# + _cell_guid="b954644f-19d5-4edc-acc8-cf943a58534b" _uuid="74d6076bd9236daf2cefd6c32b79119f6437923d"
# Voila nos données d'apprentissage
TrainingData.head()
# -

label_encoder_sex = LabelEncoder()
label_encoder_title = LabelEncoder()
label_encoder_embarked = LabelEncoder()
TrainingData['Sex'] = label_encoder_sex.fit_transform(TrainingData['Sex'].values)
TrainingData['Title'] = label_encoder_title.fit_transform(TrainingData['Sex'].values)
TrainingData['Embarked'] = label_encoder_embarked.fit_transform(TrainingData['Sex'].values)

TrainingData.head()

# + _cell_guid="6ef57b0c-9723-4f40-b8c3-e2be60784f91" _uuid="9401c53ce8a950e7aebf3fbf62faf29c5788bf65"
#On va maintenant passer du monde panda au monde numpy pour servir d'input à l'apprentissage pour cela on isole la variable cible
y = TrainingData.iloc[:, 0].values
#et le reste du dataset
X = TrainingData.iloc[:, 1:12].values

# + _cell_guid="d0b132d9-ecae-4ea0-b5f1-64fc90b8a358" _uuid="6787d0472b874437a300ac0d32e20eb3c666de09"
# Feature Scaling
scaler_x = MinMaxScaler((-1,1))
X = scaler_x.fit_transform(X)


# + _cell_guid="b768a96d-8f55-440f-b268-26fb613acc59" _uuid="dfcfa465aa94260c2db0a6c87c9dea05cc0c45de"
# On splite notre dataset d'apprentisage pour faire de la validation croisée une partie pour apprendre une partie pour regarder le score
# On utilise pas le dataset de test de kaggle notre but est de se servir du dataset d'apprentissage et de s'assurer que le modèle le généralise bien pour l'appliquer ensuite sur le dataset de test du challenge
# Prenons arbitrairement 10% du dataset en test et 90% pour l'apprentissage
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# + [markdown] _cell_guid="8fd52bae-823d-461e-83d0-cd4692e9cc52" _uuid="4b542c06467c1b212ad491671ed0c6c3ff945c2e"
# # Random Forest
#
# Les forets aléatoires sont des algorithmes souvent performants en ce qui concerne les classifications.
#
# L'idée est simple :
# * On prend dans le dataset une partie des données et une partie des variables au hasard.
# * On fait un arbre de décision sur ces données tirées aléatoirement, l'arbre de décision étant un algorithme permettant de déterminer la variable et sa valeur qui permet de séparer au mieux la population par rapport à notre variable cible le but étant de descendre aux feuilles les plus pures.
#
# ![image.png](attachment:image.png)
#
# * Puis on recommence avec une autre sous partie des données et des variables, ce qui nous fait un second arbre...
# * Du coup, plusieurs arbres aléatoires, ca nous fait une forêt...aléatoire.
# * Un individu à prédire passera dans chacun des arbres et aura pour chaque arbre une prédiction, la prédiction finale étant la pondération de chacun de nos arbres.
#
# -

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import pathlib
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

# + _cell_guid="39d76ad5-c0b0-4007-b04d-f885e67bd0d3" _uuid="8c304bf4dca9102a7c78af2c4c6298caf46a1ed9"
#Ici demandons d'avoir 20 arbres
rdmf = RandomForestClassifier(n_estimators=20)
rdmf.fit(X_train, y_train)

# + _cell_guid="2d5be6b3-51f3-4b79-af78-53e2550edc69" _uuid="861630a841959c68f0ce446f9d11c0561c514237"
#calculons le score sur le dataset d'apprentissage et sur le dataset de test (10% du dataset d'apprentissage mis de côté)
# le score étant le nombre de bonne prédiction



rdmf_score = rdmf.score(X_test, y_test)
rdmf_score_tr = rdmf.score(X_train, y_train)
print("{} % de bonnes réponses sur les données de test pour validation (résultat qu'on attendrait si on soumettait notre prédiction sur le dataset de test.csv)".format(round(rdmf_score*100)))
from sklearn.metrics import confusion_matrix
print("matrice de confusion")
confusion_matrix(y_test, rdmf.predict(X_test))
# -

# ### HyperParamètres
#
# Et voila vous avez votre premier modèle en ayant fait peu de chose vous pouvez espérer avoir sur le jeu de test Kaggle, sans comprendre d'avantage les données, on peut faire mieux en effet si l'on souhaite comprendre un peu plus les forets aléatoires et pour généraliser mieux le modèle il faut savoir que ce modèle dépend notamment de :
#
# * le critère de séparation dans l'arbre de décision qui peut etre gini ou entropy
# * la profondoeur de l'arbre de décision (on peut lui demander de faire un arbre jusqu'a avoir des feuilles totalement pures mais alors parfois représentant un individu uniquement ou lui dire de considérer qu'une feuille avec n individus ne peut plus etre splitté au risque de tomber dans du surapprentissage)
# * le nombre de variables sélectionnées
# * le nombre d'arbres constituants la forêt.
#
# On appelle ces paramètres des hyperparamètres et il s'agit pour nous de trouver les meilleurs. Pour des modèles plus compliqués, parfois il y a carrément de la modélisation pour trouver ces hyperparamètres du modèle car il s'agit d'un problèmes d'optimisation.
#
# Dans notre cas on va utiliser la puissance des machines pour essayer de trouver parmi un ensemble la meilleure combinaison en utilisant la librairie scikit learn (il va faire touner sur toutes les combinatoires et nous donner la combinaison gagnante)

# + _cell_guid="a37297a7-07c4-4b48-aadf-b3db3ef39333" _uuid="9a0c4e73306f2c932dd81897d4ff4482d0cc88b7"
# on va faire chercher par validation croisée le meilleurs modèle parmi la combinatoire des paramètres suivants 
# possible sur les random forest https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

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
# -

# Petit aparté notez que les forets aléatoires sont aussi parfois un algorithme permettant de sélectionner les variables contribuant le plus à la variable cible (Remarquez que la longueur du ticket n'est pas si mal classé !)

# +
print(gs.best_estimator_.feature_importances_)
print(TrainingData.columns[1:])
import pandas as pd
# %matplotlib inline
#do code to support model
#"data" is the X dataframe and model is the SKlearn object

feats = {} # a dict to hold feature_name: feature_importance
for feature, importance in zip(TrainingData.columns[1:], gs.best_estimator_.feature_importances_):
      feats[feature] = importance #add the name/value pair 

importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'importance'})
importances.sort_values(by='importance').plot(kind='bar', rot=45)
# -

# ### Pour faire mieux
#
# On peut difficilement faire pire car tout a été traité au plus simple :
#
# * Extraire des variables plus significatives notamment sur la situation familiale
# * améliorer le remplacement des valeurs manquantes.
# * 
# Dans tous les cas il est possible de partir de ce meme dataset et d'essayer d'autres algorithmes qui aurait un meilleur pouvoir de prédiction sur les mêmes bases. ExempleXGbooster qui souvent est un concurrent crédible des random forest.
#
# Les variables étant déjà comprises entre -1 et 1 et la variable à prédire étant 0 et 1 il est aussi possible d'appliquer un réseau de neurone avec tensorflow.
