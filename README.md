# Scraper d'Informations Juridiques sur la Succession

Ce programme utilise l'API GPT pour collecter et analyser automatiquement des informations juridiques concernant les actes de succession à partir de sites gouvernementaux français.

## Prérequis

- Python 3.8 ou supérieur
- Une clé API OpenAI

## Installation

1. Clonez ce dépôt
2. Installez les dépendances :
```bash
pip install -r requirements.txt
```
3. Créez un fichier `.env` à la racine du projet avec votre clé API OpenAI :
```
OPENAI_API_KEY=votre_clé_api_ici
```

## Utilisation

Pour lancer le programme :
```bash
python succession_scraper.py
```

Le programme va :
1. Scanner périodiquement les sites juridiques configurés
2. Utiliser GPT pour analyser le contenu et extraire les informations pertinentes
3. Sauvegarder les données dans un fichier `succession_data.json`

## Structure des données

Les données sont sauvegardées dans un fichier JSON avec la structure suivante :
```json
{
    "last_update": "timestamp",
    "clauses": {
        "clause_key": {
            "content": "contenu de la clause",
            "source": "url source",
            "date_added": "timestamp"
        }
    },
    "sources": ["liste des urls sources"]
}
```
