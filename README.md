# Système de Correspondance de Candidats

Ce projet propose une solution avancée de correspondance de candidats basée sur des techniques modernes d'analyse sémantique, de scoring multicritères et d'automatisation des processus RH. L'objectif est d'identifier efficacement les candidats les plus pertinents pour un poste donné.

## Objectif

Automatiser et optimiser le processus de sélection des candidats en exploitant l'intelligence artificielle pour une évaluation objective et complète, minimisant ainsi les biais humains.

## Structure et Fonctionnement du Projet

Le processus global du projet est orchestré par le fichier principal `main.py`, suivant ces étapes :

### 1. Extraction des Données depuis un PDF
- **Module :** `extract_data.py`  
- **Fonction :** Conversion automatique du fichier PDF décrivant le poste en Markdown, facilitant l'extraction et le traitement du texte.

### 2. Extraction Structurée via API Groq
- **Module :** `llm.py`  
- **Fonction :** Analyse du Markdown avec l'API Groq (modèle Llama-3), pour extraire les données structurées sous format JSON.

### 3. Lecture des Données Candidats
- **Module :** `csv_to_dataframe.py`  
- **Fonction :** Chargement des données candidats provenant de fichiers CSV dans des DataFrames pandas.

### 4. Nettoyage et Agrégation des Données
- **Module :** `clean_dataframes.py`  
- **Fonction :** Suppression des doublons et agrégation des informations candidats dans un DataFrame complet et propre.

### 5. Recherche Sémantique sur les Compétences
- **Module :** `semantic_search_skills.py`  
- **Fonction :** Identification des candidats pertinents en fonction des compétences requises via ChromaDB.

### 6. Prétraitement des Missions
- **Module :** `preprocess_missions.py`  
- **Fonction :** Normalisation, segmentation, lemmatisation et embeddings des missions candidats.

### 7. Recherche Sémantique sur les Missions
- **Module :** `semantic_search_missions.py`  
- **Fonction :** Recherche sémantique reliant responsabilités du poste et expériences candidats.

### 8. Reclassement Multicritères
- **Module :** `ranking_skills_missions.py`  
- **Fonction :** Calcul et combinaison des scores compétences/missions.

### 9. Fusion Scores et Informations Candidats
- **Étape :** Fusion du ranking initial avec les informations candidat prétraitées.

### 10. Score de Disponibilité
- **Module :** `compute_disponilite.py`  
- **Fonction :** Évaluation disponibilité candidats selon la durée du mandat.

### 11. Score de Compétences Linguistiques
- **Module :** `language_score.py`  
- **Fonction :** Attribution d'un score linguistique basé sur les langues exigées par l'offre.

### 12. Score Global et Sélection Finale
- **Module :** `compute_global_score.py`  
- **Fonction :** Pondération finale des scores pour identifier les 5 meilleurs candidats.

## Résultats

Les résultats sont sauvegardés automatiquement dans un fichier CSV nommé selon le poste (ex : `Scrum_top5_candidats.csv`).

## Technologies Utilisées
- Python
- pandas
- ChromaDB
- LangChain
- Groq API (Llama-3)
- NLTK, spaCy, SentenceTransformer

## Exécution du Projet

### Prérequis
- Installation des librairies (`requirements.txt`)
- Clé API Groq (`.env`)
