import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

# Charger les données
data = pd.read_csv('donnees_recommandations.csv')

# Diviser les données en variables d'entrée (X) et la variable cible (y)
X = data.drop('client_id', axis=1)
y = data['client_id']

# Diviser les données en ensemble d'entraînement et ensemble de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer le modèle de recommandation basé sur KNN
model = NearestNeighbors(n_neighbors=5, algorithm='auto')
model.fit(X_train)

# Faire des prédictions sur l'ensemble de test
distances, indices = model.kneighbors(X_test)

# Afficher les recommandations pour chaque client de l'ensemble de test
for i in range(len(X_test)):
    print("Recommandations pour le client", y_test.iloc[i], ":")
    neighbors_indices = indices[i][1:]  # Exclure le client lui-même (le plus proche voisin)
    recommendations = y_train.iloc[neighbors_indices]
    print(recommendations)
    print("-------------------")
