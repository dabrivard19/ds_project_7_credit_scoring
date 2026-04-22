
# Credit scoring app - version 10 variables

Cette version est pensée pour une démo ou une soutenance :
- l'utilisateur saisit seulement 10 variables
- les autres colonnes du dataset Home Credit sont remplies automatiquement avec des valeurs par défaut issues de `application_test.csv`

## Ajouter le modèle

Place ton fichier ici :

```
model/modele.joblib
```

## Installation

```bash
pip install -r requirements.txt
```

## Lancer l'API

```bash
uvicorn app.main:app --reload
```

## Lancer Streamlit

```bash
streamlit run ui/streamlit_app.py
```

## Tests

```bash
pytest
```

## Déploiement Railway

Le fichier `Procfile` contient la commande de démarrage de l'API.

## Attention

Cette version fonctionne bien pour une démonstration. Pour la production, il faut idéalement exposer exactement les mêmes variables et le même prétraitement que ceux utilisés pendant l'entraînement du modèle.
