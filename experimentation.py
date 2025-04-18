# %%
from docling.document_converter import DocumentConverter
from langchain_core.output_parsers import JsonOutputParser
from groq import Groq
from dotenv import load_dotenv
import os
import requests 
import chromadb
import pandas as pd
import uuid
# %%
# Initialize the DocumentConverter
converter = DocumentConverter()

# Path to the PDF file in the 'ressources' folder
pdf_file_path = r"C:\Users\samso\OneDrive - HEC Montréal\Documents\Personnal projects\ChatBot\RoboStaff\ressources\Scrum.pdf"

# Convert the PDF file
result = converter.convert(pdf_file_path)

# Extract the document object
document = result.document

# Export the document to Markdown and JSON formats
markdown_output = document.export_to_markdown()
json_output = document.export_to_dict()

# Print the Markdown output
print(markdown_output)
# %%
# %% Charger le token d'API
# Load environment variables from the .env file
# Construct the path to the .env file relative to this script
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)
# Load the API key from the .env file
api_key = os.getenv("GROQ_API_Token")
if not api_key:
    print("Current Working Directory:", os.getcwd())
    raise ValueError("API key not found. Please set GROQ_API_KEY in your .env file.")
groq = Groq(api_key=api_key)
# %%
prompt = f"""
        ### TEXTE EXTRAPOLÉ DU CV:
        {markdown_output}
        ### INSTRUCTIONS:
        Le texte provient d'un CV. Votre mission est d'extraire les informations suivantes et de les renvoyer sous forme d'un objet JSON en respectant la structure ci-dessous :
        
        - mandate_duration : la durée du mandat (exemple : "10 mois (renouvelable)")
        - mandate_type : le type de mandat (exemple : "Temps plein")
        - experience : l'expérience professionnelle mentionnée (exemple : "Minimum 5 ans d’expérience en analyse de données")
        - responsibilities : les responsabilités mentionnées (exemple : "Analyser les données et les transformer en informations utiles")
        - tools : les outils à maîtriser (exemple : "SQL, Power BI, Databricks, etc.")
        - languages : les langues parlées (exemple : "Français, Anglais")
        - catégorie_poste : seulement 2 catégories du poste possibles , soit "plus technique" ou "plus gestion"

        Only return the valid JSON.
        ### VALID JSON (NO PREAMBLE):
        """

chat_completion = groq.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
        temperature=0.5
    )
content = chat_completion.choices[0].message.content
# %%
json_parser = JsonOutputParser()
json_res = json_parser.parse(content)
# %%
json_res
# %% Data exploration for HCK_HEC_USER
# Load the HCK_HEC_USER data
file_path_user = r"C:\Users\samso\OneDrive - HEC Montréal\Documents\Personnal projects\ChatBot\RoboStaff\ressources\HCK_HEC_USER.csv"
df_user = pd.read_csv(file_path_user)
# %%Explore the user data
df_user.shape # 119 rows and 3 columns

# %% Data exploration for HCK_HEC_XP
file_path_xp = r"C:\Users\samso\OneDrive - HEC Montréal\Documents\Personnal projects\ChatBot\RoboStaff\ressources\HCK_HEC_XP.csv"
df_xp = pd.read_csv(file_path_xp)
# %%Explore the xp data
df_xp.shape # 1853 rows and 2 columns

# %%
df_xp.value_counts()
# %%
df_xp.USER_ID.nunique() # 114 UNIQUE USERS
# %%
df_xp[(df_xp.USER_ID == 2433099) & (df_xp.MISSION_DSC == "Gérer le budget et produire les rapports d'avancements.")] # Repetition of the same mission for same user
# %% Data exploration for HCK_HEC_SKILL
file_path_skill = r"C:\Users\samso\OneDrive - HEC Montréal\Documents\Personnal projects\ChatBot\RoboStaff\ressources\HCK_HEC_SKILLS.csv"
df_skill = pd.read_csv(file_path_skill)
# %%Explore the skill data
df_skill.shape # 3411 rows and 2 columns

# %%
df_skill.head()
# %%
df_skill.drop_duplicates(inplace=True)
# %%
df_skill.shape # 987 rows and 2 columns after dropping duplicates
# %%
df_skill["LEVEL_VAL"].median() #70
df_skill["LEVEL_VAL"].fillna(df_skill["LEVEL_VAL"].median(), inplace=True) # Replace NaN with median for numeric columns
# %% Data exploration for the HCK_HEC_LANG
file_path_lang = r"C:\Users\samso\OneDrive - HEC Montréal\Documents\Personnal projects\ChatBot\RoboStaff\ressources\HCK_HEC_LANG.csv"
df_lang = pd.read_csv(file_path_lang)

# %%
df_lang.shape #217 rows and 3 columns
# %%
df_lang.USER_ID.nunique() # 112 UNIQUE USERS
# %%
df_lang.LANGUAGE_SKILL_DSC.value_counts() #English    113  | French     104
# %%Data exploration for the HCK_HEC_STAFFING
file_path_staffing = r"C:\Users\samso\OneDrive - HEC Montréal\Documents\Personnal projects\ChatBot\RoboStaff\ressources\HCK_HEC_STAFFING.csv"
df_staffing = pd.read_csv(file_path_staffing)

# %%
df_staffing.shape # 119 rows and 13 columns
# %% ## DATA CLEANING
# %%
# Drop the duplicates in the xp dataframe
df_xp.drop_duplicates(inplace=True)
# %%
df_xp.shape
# %%Join the experience descriptions into a single string (or list) per user.
#Group the xp dataframe by USER_ID and MISSION_DSC
df_xp_grouped = df_xp.groupby("USER_ID")["MISSION_DSC"].apply(lambda x: ". ".join(x)).reset_index()
#Rename the column
df_xp_grouped.rename(columns={"MISSION_DSC":"MISSIONS"}, inplace=True)
# %%
df_xp_grouped.shape # 114 rows and 2 columns
# %%
#Find users that are not in the df_xp_grouped dataframe but are in the df_user dataframe
missing_users = df_user[~df_user.USER_ID.isin(df_xp_grouped.USER_ID)]
# %% Aggregate HCK_HEC_SKILL: combine skill description and level per USER_ID
df_skill_grouped = df_skill.groupby('USER_ID').apply(
    lambda df: ', '.join(df['SKILLS_DSC'] + ' (' + df['LEVEL_VAL'].astype(str) + ')')
).reset_index(name='skills_info')
# %%
df_skill_grouped.shape # 93 columns and 2 rows
# %%
#Find users that are not in the df_skill_grouped dataframe but are in the df_user dataframe
missing_users_skill = df_user[~df_user.USER_ID.isin(df_skill_grouped.USER_ID)] # 26 missing users
# %%Aggregate HCK_HEC_LANG: combine language description and level per USER_ID
df_lang_agg = df_lang.groupby('USER_ID').apply(
    lambda df: ', '.join(df['LANGUAGE_SKILL_DSC'] + ' (' + df['LANGUAGE_SKILL_LVL'].astype(str) + ')')
).reset_index(name='languages_info')

# %% JOIN ALL THE AGGREGATED DATAFRAMES
df_completed_user = df_user.merge(df_xp_grouped, on="USER_ID", how="left") #Merge experience
df_completed_user = df_completed_user.merge(df_skill_grouped, on="USER_ID", how="left") #Merge skills
df_completed_user = df_completed_user.merge(df_lang_agg, on="USER_ID", how="left") #Merge language
df_completed_user = df_completed_user.merge(df_staffing, on="USER_ID", how="left") #Merge staffing
# %%
df_completed_user["skills_info"] = df_completed_user["skills_info"].fillna("")
df_completed_user["MISSIONS"] = df_completed_user["MISSIONS"].fillna("")
df_completed_user["languages_info"] = df_completed_user["languages_info"].fillna("")
df_completed_user["ANNEES_XP"] = df_completed_user["ANNEES_XP"].fillna(0)  # Replace NaN with 0 for numeric columns
# %%
#--------------------------------------------------------------------------------
# TEST DURÉE REGEX
#--------------------------------------------------------------------------------
# # Exemples d'utilisation :
# exemples = [
#     "10 mois (renouvelable)",
#     "expectations: 2 years as soon as possible",
#     "3 months",
#     "5 jours",
#     "1 year",
#     "7 days"
# ]
# for texte in exemples:
#     match = pattern.search(texte)
#     if match:
#         duration = int(match.group("duration"))
#         period = match.group("period").lower()
#         print(f"Texte: '{texte}' -> Durée: {duration}, Période: {period}")
#     else:
#         print(f"Texte: '{texte}' -> Pas de correspondance trouvée")
#---------------------------------------------------------------------------------
import re

pattern = re.compile(r"(?i)(?P<duration>\d+)\s*(?P<period>jours?|days?|mois|months?|ann[ée]e?s?|years?)")

mandate_duration = json_res["mandate_duration"]
match = pattern.search(mandate_duration)
duration = int(match.group("duration"))
period = match.group("period").lower()
# %%
client = chromadb.Client()
collection = client.get_or_create_collection(name="staff_matching_test")

if not collection.count():
    for _, row in df_skill.iterrows():
        collection.add(documents=row["SKILLS_DSC"],
                       metadatas={"User_ID":row["USER_ID"],
                                  "SKILLS_DSC":row["SKILLS_DSC"],
                                  "LEVEL_VAL":row["LEVEL_VAL"]},
                        ids=[str(uuid.uuid4())])
# %%

results_only_skills  = collection.query(query_texts=json_res["tools"], n_results=5,include=["distances", "metadatas"])

# %%
# Process the results (assuming results is a dictionary containing your IDs and scores)
top_candidate_ids = results_only_skills["metadatas"][0]
# %%
for tool in json_res["tools"]:
    results = collection.query(query_texts=tool, n_results=5, include=["distances", "metadatas"])
# %%
result = collection.query(query_texts=json_res["tools"][0], n_results=round(df_skill.shape[0]/6), include=["distances", "metadatas"])
type(result)
# %%Calculate the mean distance in the list of distances
mean_distance = sum(result["distances"][0])/len(result["distances"][0])
# %% List to a dataframe
df_distances = pd.DataFrame(result["distances"][0])
mean_distance = float(df_distances.describe().loc["50%"])

import matplotlib.pyplot as plt
# Créer un histogramme
plt.hist(df_distances, bins=20, color='blue', edgecolor='black')
plt.title("Distribution des distances (0.2 - 0.8)")
plt.xlabel("Valeurs")
plt.ylabel("Fréquence")
plt.show()
# %%
df_distances[df_distances[0] < mean_distance].shape
result["metadatas"]
# %%
df_query_results = pd.DataFrame(result["metadatas"][0])
df_query_results_sorted = df_query_results.head(30).sort_values(by="LEVEL_VAL", ascending=False)

result = collection.query(query_texts=json_res["tools"][3], n_results=round(df_skill.shape[0]/6), include=["distances", "metadatas"])

df_tools = pd.DataFrame(json_res["tools"])

# Iterate over each row in df_test and perform the collection.query function
query_results = []

for _, row in df_tools.iterrows():
    tool = row[0]  # Assuming the tool name is in the first column
    result = collection.query(query_texts=tool, n_results=round(df_skill.shape[0]/6), include=["distances", "metadatas"])
    query_results.append({
        "tool": tool,
        "results": result
    })

# Convert the query results into a DataFrame for further analysis
df_query_results = pd.DataFrame(query_results)
list_tools = json_res["tools"]
df_query_results["results"][0]["metadatas"][0]
# %%
# Create a dictionary to store the user IDs for each tool
skill_to_user_ids = {tool: [] for tool in list_tools}

for i, row in df_query_results.iterrows():
    # Get the current tool from list_tools based on index
    current_tool = list_tools[i]
    # Extract metadata as a DataFrame for this query result
    df_query_skills = pd.DataFrame(df_query_results["results"].iloc[i]["metadatas"][0])
    
    # Calculate the number of skills to choose (10% of available skills)
    nb_skills_chosen = round(df_query_skills.shape[0] * 0.10)
    
    # Select the top 10% skills sorted by LEVEL_VAL in descending order
    df_query_skills_selected = df_query_skills.sort_values(by="LEVEL_VAL", ascending=False).head(nb_skills_chosen)
    
    # For each row, add the USER_ID to the dictionary for the corresponding skill
    for _, skill_row in df_query_skills_selected.iterrows():
        # Directly use the skill as key and append the USER_ID
        skill_to_user_ids[current_tool].append(skill_row["User_ID"])

print(skill_to_user_ids) # Dictionary with tools as keys and lists of USER_IDs as values
# %%
from collections import Counter

def get_user_id_occurrences_sorted(skill_to_user_ids):
    """
    Given a dictionary where keys are skills and values are lists of user IDs,
    return a dictionary sorted in descending order by the number of times each user ID appears.
    The keys are the user IDs and the values are the counts.
    """
    user_counter = Counter()
    for user_ids in skill_to_user_ids.values():
        user_counter.update(user_ids)
    
    # Sort the counter items by count in descending order and convert to a dictionary
    sorted_user_counts = dict(sorted(user_counter.items(), key=lambda item: item[1], reverse=True))
    return sorted_user_counts

common_user_ids = get_user_id_occurrences_sorted(skill_to_user_ids)
print(common_user_ids)
# %%
#size of the dictionnary
len(common_user_ids) #76 users that are potential good candidates
# %%
# PREMIÈRE ÉTAPE DE PRETRAITEMENT DU TEXTE DE MISSIONS
df_completed_user["MISSIONS"]

import re

# Exemple : DataFrame df_completed_user avec la colonne "MISSIONS"
# df_completed_user = pd.read_csv("votre_fichier.csv")  # Chargement éventuel

def preprocess_step1(text):
    """
    Étape 1 : Normalisation de base du texte
    - Convertit le texte en minuscules pour uniformiser la casse.
    - Supprime les espaces en trop (début/fin de chaîne, espaces multiples).
    - Préserve la ponctuation importante (points, etc.) pour garder la structure.
    """
    if not isinstance(text, str):
        # Gestion de cas non-string, par exemple NaN
        text = str(text)
    
    # Convertir en minuscules
    text = text.lower()

    # Supprimer les espaces de début/fin
    text = text.strip()

    # Remplacer plusieurs espaces par un seul
    text = re.sub(r"\s+", " ", text)
    
    return text

# Application de l'étape 1 de prétraitement sur la colonne MISSIONS
df_completed_user["MISSIONS"] = df_completed_user["MISSIONS"].apply(preprocess_step1)
#%%
# DEUXIÈME ÉTAPE DE PRETRAITEMENT DU TEXTE DE MISSIONS
import nltk

# Vérifier si la ressource 'punkt_tab' pour le français est disponible, sinon la télécharger
try:
    nltk.data.find('tokenizers/punkt_tab/french')
except LookupError:
    nltk.download('punkt_tab')

from nltk.tokenize import sent_tokenize

def preprocess_step2(text):
    """
    Étape 2 : Segmentation en phrases
    - Segmente le texte en phrases, en s'appuyant sur la ponctuation.
    - Retourne une liste de phrases, ce qui permet de conserver un certain contexte
      pour chaque phrase lors des futures analyses sémantiques.
    """
    # Vérifier que l'entrée est bien une chaîne de caractères
    if not isinstance(text, str):
        text = str(text)
    
    # Utilisation du tokenizer NLTK (en supposant un texte principalement en français)
    sentences = sent_tokenize(text, language='french')
    
    return sentences

# Application de l'étape 2 sur la colonne MISSIONS (après la première étape de prétraitement)
df_completed_user["MISSIONS_sentences"] = df_completed_user["MISSIONS"].apply(preprocess_step2)
# %%
# TROISIÈME ÉTAPE DE PRETRAITEMENT DU TEXTE DE MISSIONS
import spacy

# Charger le modèle spaCy pour le français de manière dynamique
try:
    nlp = spacy.load("fr_core_news_sm")
except Exception as e:
    import spacy.cli
    spacy.cli.download("fr_core_news_sm")
    nlp = spacy.load("fr_core_news_sm")

def lemmatize_sentence(sentence):
    """
    Lemmatisation d'une phrase :
    - Traite la phrase avec spaCy pour récupérer la forme de base (lemma) de chaque mot.
    - Retourne une nouvelle phrase constituée des lemmes, tout en gardant l'ordre des mots.
    """
    doc = nlp(sentence)
    lemmatized_tokens = [token.lemma_ for token in doc]
    # Reconstituer la phrase à partir des lemmes
    return " ".join(lemmatized_tokens)

def preprocess_step3(sentences_list):
    """
    Étape 3 : Lemmatisation de chaque phrase dans la liste
    - Prend en entrée une liste de phrases (issues de la segmentation précédente).
    - Applique la fonction de lemmatisation à chacune des phrases.
    - Retourne une liste de phrases lemmatisées, en conservant l'ordre pour préserver le contexte.
    """
    return [lemmatize_sentence(sentence) for sentence in sentences_list]

# Application de l'étape 3 sur la colonne "MISSIONS_sentences"
df_completed_user["MISSIONS_lemmatized"] = df_completed_user["MISSIONS_sentences"].apply(preprocess_step3)
#%%
# QUATRIÈME ÉTAPE DE PRETRAITEMENT DU TEXTE DE MISSIONS QUI CONSISTE À CONVERTIR LA LISTE DE PHRASES LEMMATISÉES EN EMBEDDINGS NUMÉRIQUES.
from sentence_transformers import SentenceTransformer

# Charger dynamiquement un modèle de sentence embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')  # Ce modèle est performant et léger

def compute_embeddings(sentences_list):
    """
    Étape 4 : Vectorisation des missions lemmatisées
    - Prend en entrée une liste de phrases lemmatisées.
    - Calcule pour chaque phrase un embedding numérique à l'aide du modèle.
    - Retourne une liste d'embeddings, en gardant chaque phrase comme unité contextuelle.
    """
    # Si la liste est vide ou contient des valeurs invalides, on retourne une liste vide
    if not sentences_list or not isinstance(sentences_list, list):
        return []
    embeddings = model.encode(sentences_list, convert_to_tensor=False)
    return embeddings

# Application de la fonction sur la colonne "MISSIONS_lemmatized"
df_completed_user["MISSIONS_embeddings"] = df_completed_user["MISSIONS_lemmatized"].apply(compute_embeddings)
# %%
# CINQUIÈME ÉTAPE DE PRÉTRAITEMENT DU TEXTE DE MISSIONS QUI CONSISTE À CALCULER LES SIMILARITÉS ENTRE LES MISSIONS DES UTILISATEURS ET LES RESPONSABILITÉS DU POSTE.
from collections import defaultdict

# Création du client ChromaDB et récupération ou création de la collection
client = chromadb.Client()
collection_mission = client.get_or_create_collection(name="missions_candidates")

# Ajout des missions des candidats dans la collection
# Chaque document correspond à une phrase de mission prétraitée (lemmatisée)
for idx, row in df_completed_user.iterrows():
    user_id = row["USER_ID"]
    mission_sentences = row["MISSIONS_lemmatized"]
    mission_embeddings = row["MISSIONS_embeddings"]
    for sentence, embedding in zip(mission_sentences, mission_embeddings):
        # Conversion de l'embedding (numpy array) en liste Python
        embedding_list = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
        collection_mission.add(
            documents=[sentence],
            embeddings=[embedding_list],
            metadatas={"User_ID": user_id},
            ids=[str(uuid.uuid4())]
        )

# Définition des responsabilités extraites du JSON du poste
responsibilities = json_res["responsibilities"]

# Dictionnaire qui contiendra pour chaque responsabilité le top 10% des candidats
responsibility_top_candidates = {}

# Pour chaque responsabilité dans le JSON
for resp in responsibilities:
    query_result = collection_mission.query(
        query_texts=[resp],
        n_results=100,  # Ajustez ce paramètre en fonction de la taille de la collection
        include=["distances", "metadatas"]
    )
    
    # Pour chaque résultat, conserver la meilleure (la plus petite) distance par candidat
    candidate_best_distance = {}
    for i, metadata in enumerate(query_result["metadatas"][0]):
        user_id = metadata["User_ID"]
        distance = query_result["distances"][0][i]
        if user_id in candidate_best_distance:
            candidate_best_distance[user_id] = min(candidate_best_distance[user_id], distance)
        else:
            candidate_best_distance[user_id] = distance
    
    # Trie des candidats par distance (croissante = meilleure similarité)
    sorted_candidates = sorted(candidate_best_distance.items(), key=lambda x: x[1])
    
    # Calcul du nombre de candidats à sélectionner (top 30%)
    num_candidates = len(sorted_candidates)
    top_n = max(1, int(num_candidates * 0.3))
    
    # Liste des User_ID des top candidats pour cette responsabilité
    top_candidate_ids = [user_id for user_id, _ in sorted_candidates[:top_n]]
    
    # Stockage dans le dictionnaire
    responsibility_top_candidates[resp] = top_candidate_ids

print("Dictionnaire des top candidats par responsabilité :")
print(responsibility_top_candidates)

# %% STRATEGIE POUR TROUVER LES CANDIDATS DANS LES DEUX DICTIONNAIRES ET ÉTABLIR UN RANKING GLOBAL
# %%
# 1. Pour le dictionnaire des responsabilités

# a. Comptage simple (count)
resp_count = {}
for resp, candidates in responsibility_top_candidates.items():
    for cand in candidates:
        resp_count[cand] = resp_count.get(cand, 0) + 1

# b. Score basé sur la position
resp_pos_score = {}
for resp, candidates in responsibility_top_candidates.items():
    n = len(candidates)
    for idx, cand in enumerate(candidates):
        score = (n - idx) / n  # meilleur score = 1 pour le premier candidat, 1/n pour le dernier
        resp_pos_score[cand] = resp_pos_score.get(cand, 0) + score

# c. Score final pour les responsabilités : 40% du count + 70% du score de position
final_resp_score = {}
for cand in resp_count:
    final_resp_score[cand] = 0.3 * resp_count[cand] + 0.7 * resp_pos_score.get(cand, 0)


# 2. Pour le dictionnaire des skills/ou outils

# a. Comptage simple (count)
skills_count = {}
for skill, candidates in skill_to_user_ids.items():
    for cand in candidates:
        skills_count[cand] = skills_count.get(cand, 0) + 1

# b. Score basé sur la position
skills_pos_score = {}
for skill, candidates in skill_to_user_ids.items():
    n = len(candidates)
    for idx, cand in enumerate(candidates):
        score = (n - idx) / n
        skills_pos_score[cand] = skills_pos_score.get(cand, 0) + score

# c. Score final pour les skills : 30% du count + 70% du score de position
final_skills_score = {}
for cand in skills_count:
    final_skills_score[cand] = 0.3 * skills_count[cand] + 0.7* skills_pos_score.get(cand, 0)


# 3. Ne garder que les candidats présents dans les deux dimensions
common_candidates = set(final_resp_score.keys()).intersection(set(final_skills_score.keys()))

# 4. Calcul du score global pondéré :
#    40 % pour les responsabilités et 60 % pour les skills.
global_scores = {}
for cand in common_candidates:
    global_scores[cand] = 0.3 * final_resp_score[cand] + 0.7 * final_skills_score[cand]

# --------------------------
# 4. Création du DataFrame classé

# Conversion du dictionnaire global_scores en DataFrame
df_ranked = pd.DataFrame(list(global_scores.items()), columns=["USER_ID", "SCORE_SKILLS_MISSIONS"])

# Classement par score décroissant (le meilleur score en premier)
df_ranked = df_ranked.sort_values(by="SCORE_SKILLS_MISSIONS", ascending=False).reset_index(drop=True)

print("Ranking global des candidats :")
print(df_ranked)

df_final = pd.merge(df_ranked[['USER_ID', 'SCORE_SKILLS_MISSIONS']],
                        df_completed_user,
                        on='USER_ID',
                        how='inner')

