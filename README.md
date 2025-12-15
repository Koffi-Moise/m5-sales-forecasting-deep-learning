# M5 Sales Forecasting with Deep Learning

Projet de prévision de ventes utilisant des modèles de Deep Learning dans le cadre du module **Applied Deep Learning**.

## À propos du projet

Ce projet implémente et compare trois architectures de réseaux de neurones pour prédire les ventes de produits retail à partir du dataset M5 Forecasting Competition (Walmart).

### Modèles implémentés :
- **MLP (Multilayer Perceptron)** - Réseau de neurones dense
- **SimpleRNN** - Réseau récurrent simple
- **GRU (Gated Recurrent Unit)** - Réseau récurrent avec mécanisme de portes

## Objectifs

- Prétraiter et normaliser des données de séries temporelles de ventes
- Créer des séquences temporelles pour l'apprentissage supervisé
- Entraîner et comparer différentes architectures de Deep Learning
- Évaluer les performances avec plusieurs métriques (MAE, RMSE, MAPE, R²)

## Dataset

**M5 Forecasting - Accuracy** (Kaggle)
- `calendar.csv` : Informations calendaires et événements
- `sales_train_validation.csv` : Historique des ventes
- `sell_prices.csv` : Prix de vente
- `sample_submission.csv` : Format de soumission

🔗 [Télécharger le dataset](https://www.kaggle.com/c/m5-forecasting-accuracy/data)

## Installation

```bash
# Cloner le dépôt
git clone https://github.com/votre-username/m5-sales-forecasting-deep-learning.git
cd m5-sales-forecasting-deep-learning

# Installer les dépendances
pip install -r requirements.txt
```

### Dépendances principales
```
numpy
pandas
scikit-learn
tensorflow
keras
matplotlib
```

## Utilisation

1. **Télécharger les données** depuis Kaggle et placer les fichiers CSV dans le dossier `data/m5/`

2. **Modifier le chemin des données** dans le script :
```python
INPUT_DIR = 'votre/chemin/vers/data/m5/'
```

3. **Exécuter le script** :
```bash
python m5_forecasting.py
```

## Résultats

Le projet compare les performances des trois modèles sur 10 produits sélectionnés aléatoirement :

### Métriques d'évaluation
| Modèle | MAE | RMSE | MAPE | R² |
|--------|-----|------|------|-----|
| **SimpleRNN** | ✅ Meilleur | ✅ | ✅ | ✅ |
| GRU | Bon | Bon | Bon | Bon |
| MLP | Moins performant | - | - | - |

### Conclusions
- **SimpleRNN** est le modèle le plus performant pour ce dataset
- **GRU** offre des performances proches mais avec plus de complexité
- **MLP** ne capture pas efficacement la dynamique temporelle des ventes

## Architecture technique

### Prétraitement
- Downcast des types de données pour optimiser la mémoire
- Normalisation MinMax (0-1)
- Création de séquences temporelles (timesteps = 14 jours)
- Feature engineering : événements calendaires

### Hyperparamètres
- **Timesteps** : 14 jours
- **Batch size** : 44
- **Epochs** : 32
- **Dropout** : 0.2
- **Optimizer** : Adam
- **Loss** : Mean Squared Error

## Structure du code

```
m5-sales-forecasting-deep-learning/
│
├── m5_forecasting.py          # Script principal
├── requirements.txt           # Dépendances
├── README.md                  # Documentation
│
└── data/
    └── m5/                    # Dataset (à télécharger)
        ├── calendar.csv
        ├── sales_train_validation.csv
        ├── sell_prices.csv
        └── sample_submission.csv
```

## Contexte académique

**Module** : Applied Deep Learning  
**Objectif** : Maîtriser les architectures de réseaux de neurones pour les séries temporelles  
**Compétences développées** :
- Preprocessing de données temporelles à grande échelle
- Implémentation de modèles séquentiels avec Keras
- Évaluation comparative de modèles
- Optimisation des performances

## Améliorations possibles

- [ ] Implémenter LSTM pour comparaison
- [ ] Ajouter l'attention mechanism
- [ ] Hyperparameter tuning avec Grid Search
- [ ] Validation croisée temporelle
- [ ] Prédictions multi-horizons (28 jours)
- [ ] Visualisations interactives des prédictions

## Auteur

**AKAKPO Koffi Moïse**  
Module Applied Deep Learning  
