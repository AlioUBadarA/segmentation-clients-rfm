# Segmentation clients par RFM

Projet pédagogique de Machine Learning non supervisé : regrouper les clients d'une boutique en ligne selon leur comportement d'achat, afin de personnaliser les actions marketing.

**Application en ligne : [ouvrir l'application](https://segmentation-clients-rfm-kfc5brwnnxt5xu4mn2djhj.streamlit.app)**

## Problème métier

Une entreprise e-commerce veut savoir qui sont ses meilleurs clients, lesquels risquent de partir et lesquels viennent d'arriver, pour adapter ses offres. Ce projet segmente les clients avec la méthode RFM :

- **Récence (R)** : nombre de jours depuis le dernier achat
- **Fréquence (F)** : nombre de commandes
- **Montant (M)** : total dépensé

## Données

- Jeu de données : **Online Retail** (UCI Machine Learning Repository)
- Transactions d'une boutique en ligne britannique entre décembre 2010 et décembre 2011
- 541 909 lignes brutes, 392 692 lignes après nettoyage, **4 338 clients**
- Les données ne sont pas incluses dans ce dépôt : [lien de téléchargement](https://archive.ics.uci.edu/dataset/352/online+retail)

## Méthodologie

1. **Nettoyage** : suppression des lignes sans identifiant client, des annulations de facture, des quantités et prix négatifs ou nuls, et des doublons
2. **Calcul RFM** : une ligne par client
3. **Transformation** : logarithme (`log1p`) pour réduire l'asymétrie, puis `StandardScaler`
4. **Choix du nombre de clusters** : méthode du coude, score de silhouette, indice de Davies-Bouldin et de Calinski-Harabasz ; k = 4 retenu pour sa lisibilité métier
5. **Modèle** : K-Means, intégré dans un `Pipeline` scikit-learn (log, normalisation, K-Means) sauvegardé en `.pkl`
6. **Contrôle de stabilité** : K-Means ajusté sur 20 sous-échantillons de 80 % des clients, comparé à la partition de référence avec l'indice de Rand ajusté
7. **Application** : interface Streamlit

## Résultats : 4 segments

| Segment                  | Part des clients | Récence moy. | Fréquence moy. | Montant moy. |
|--------------------------|------------------|--------------|----------------|--------------|
| Clients VIP              | 16,4 %           | 12 j         | 13,8           | 8 088 £      |
| Clients fidèles          | 26,9 %           | 72 j         | 4,1            | 1 802 £      |
| Nouveaux clients         | 19,3 %           | 18 j         | 2,2            | 557 £        |
| Clients à risque (churn) | 37,4 %           | 182 j        | 1,3            | 341 £        |

Le modèle livré (`modele_pipeline.pkl`) est celui entraîné dans le notebook `EDA.ipynb`. Les métriques de qualité (silhouette, Davies-Bouldin, stabilité) et la part du chiffre d'affaires par segment sont produites en relançant `train.py` sur les données (voir plus bas).

## Application Streamlit

Quatre fonctions :

- **Simuler un client** : saisir R, F, M, obtenir le segment et l'action marketing recommandée
- **Importer un fichier RFM** : CSV avec les colonnes Recency, Frequency, Monetary (CustomerID facultatif), avec contrôle des valeurs manquantes, non numériques et négatives
- **Importer des transactions** : CSV ou Excel au format Online Retail ; le RFM est calculé automatiquement
- **Qualité du modèle** : onglet affiché lorsque `reports/metrics.json` existe (produit par `train.py`)

Les résultats comprennent le profil de chaque segment (effectif, part des clients, part du chiffre d'affaires, action recommandée), des graphiques, un filtre par segment et l'export CSV.

Un client qui n'a passé qu'une seule commande, d'un montant élevé, est signalé comme **profil atypique** : son segment est à confirmer avant toute action. Le seuil est une heuristique (`ATYPICAL_Z_MONETARY` dans `rfm_core.py`) à ajuster selon les données.

[![Simulation d'un client](https://github.com/khadijahodonou97-lgtm/segmentation-clients-rfm/raw/main/captures/app_simulation_churn.png)](https://github.com/khadijahodonou97-lgtm/segmentation-clients-rfm/blob/main/captures/app_simulation_churn.png) [![Import d'un fichier CSV](https://github.com/khadijahodonou97-lgtm/segmentation-clients-rfm/raw/main/captures/app_import_csv.png)](https://github.com/khadijahodonou97-lgtm/segmentation-clients-rfm/blob/main/captures/app_import_csv.png)

## Structure du dépôt

| Fichier                  | Rôle                                                              |
|--------------------------|-------------------------------------------------------------------|
| `rfm_core.py`            | Nettoyage, RFM, pipeline, choix de k, stabilité, nommage, validation |
| `train.py`               | Entraînement reproductible en ligne de commande                   |
| `app.py`                 | Application Streamlit                                             |
| `tests/`                 | Tests unitaires (pytest)                                          |
| `EDA.ipynb`              | Exploration et construction initiale du modèle (Colab)            |
| `modele_pipeline.pkl`    | Modèle livré (pipeline et noms des segments)                      |

## Lancer le projet

```
pip install -r requirements.txt
streamlit run app.py
```

## Réentraîner le modèle

Télécharger le fichier `Online Retail.xlsx` (lien plus haut), puis :

```
pip install -r requirements-dev.txt
python train.py --data "Online Retail.xlsx" --k 4 --output-dir .
```

Le script écrit `modele_pipeline.pkl`, `rfm_segments.csv` et, dans `reports/`, `metrics.json`, `selection_k.csv` et `selection_k.png`. Les noms de segments sont attribués automatiquement d'après les moyennes RFM de chaque cluster (règle décrite dans `rfm_core.name_segments`).

## Tests

```
pip install -r requirements-dev.txt
pytest
```

Les tests couvrent le nettoyage, le calcul RFM, le pipeline, le nommage des segments, la validation des fichiers, l'entraînement de bout en bout sur données synthétiques et l'application (simulation et imports). Ils s'exécutent aussi à chaque push via GitHub Actions.

## Limites

- Projet pédagogique : les segments sont des profils statistiques, pas une décision commerciale validée.
- Les noms des segments sont déduits des moyennes de chaque cluster, par une règle fixe valable pour k = 4.
- Un achat exceptionnel isolé (par exemple un client avec une seule commande très élevée) peut être classé dans un segment qui ne reflète pas son vrai comportement. L'application le signale, mais le modèle n'est pas corrigé.
- Le modèle est entraîné sur une seule année de données d'une entreprise britannique.
- Le fichier `.pkl` doit être chargé avec la même version de scikit-learn (1.6.1).

## Pistes d'amélioration

- Comparer avec un scoring RFM par quantiles (1 à 5 sur chaque axe) et avec d'autres algorithmes (GMM, clustering hiérarchique)
- Ajouter des variables : panier moyen, diversité des produits, ancienneté, pays, retours
- Valider sur une période de test (entraîner sur une période, segmenter la suivante)
- Estimer la valeur vie client (CLV) et une probabilité de churn

## Auteure

Khadidiatou Lucia Sy Hodonou
