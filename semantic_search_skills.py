import pandas as pd
import chromadb
import uuid

def compute_skill_to_user_ids(df_skill, json_res):
    """
    Traite le DataFrame des skills et le JSON contenant les outils afin de créer un dictionnaire
    associant chaque outil à la liste des USER_ID correspondants (top 10% sélectionnés).
    
    Args:
        df_skill (pd.DataFrame): DataFrame contenant les compétences, avec les colonnes "USER_ID",
                                 "SKILLS_DSC" et "LEVEL_VAL".
        json_res (dict): Dictionnaire JSON contenant au moins la clé "tools" (liste des outils).
    
    Returns:
        dict: Dictionnaire où chaque clé est un outil et chaque valeur est la liste des USER_ID
              sélectionnés pour cet outil.
    """
    # Créer un client en mémoire sans persistance sur disque
    client = chromadb.Client()
    
    # Créer une collection pour stocker les compétences
    collection_skills = client.get_or_create_collection(name="skills_candidates")
    
    # Vérifier si la collection est vide avant d'ajouter les compétences
    if collection_skills.count() == 0:
        # Suppression des doublons dans le DataFrame des compétences
        df_skill.drop_duplicates(inplace=True)
        
        # Ajouter les compétences au vectorstore seulement si nécessaire
        print("Ajout des compétences à la base vectorielle en mémoire...")
        for _, row in df_skill.iterrows():
            collection_skills.add(
                documents=[row["SKILLS_DSC"]],
                metadatas={
                    "User_ID": row["USER_ID"],
                    "SKILLS_DSC": row["SKILLS_DSC"],
                    "LEVEL_VAL": row["LEVEL_VAL"]
                },
                ids=[str(uuid.uuid4())]
            )
    else:
        print(f"Utilisation de la collection existante avec {collection_skills.count()} compétences.")
            
    # Liste des outils extraits du JSON
    list_tools = json_res["tools"]
    
    # Initialiser une liste pour stocker les résultats de requête
    query_results = []
    
    # Pour chaque outil du JSON, interroger la collection et stocker le résultat
    for tool in list_tools:
        result = collection_skills.query(
            query_texts=[tool],
            n_results=round(df_skill.shape[0] / 6),  # Nombre de résultats à récupérer
            include=["distances", "metadatas"]
        )
        query_results.append({
            "tool": tool,
            "results": result
        })
    
    # Convert the query results into a DataFrame for further analysis
    df_query_results = pd.DataFrame(query_results)
    
    # Créer un dictionnaire pour stocker les USER_ID pour chaque outil
    skill_to_user_ids = {tool: [] for tool in list_tools}
    
    for i, row in df_query_results.iterrows():
        # Get the current tool from list_tools based on index
        current_tool = list_tools[i]
        # Extract metadata as a DataFrame for this query result
        df_query_skills = pd.DataFrame(df_query_results["results"].iloc[i]["metadatas"][0])
    
        # Calculate the number of skills to choose (10% of available skills)
        nb_skills_chosen = round(df_query_skills.shape[0] * 0.10)
    
        # Select the top 10% skills sorted by LEVEL_VAL in descending order
        df_query_skills_selected = df_query_skills.head(nb_skills_chosen)
        df_query_skills_sorted = df_query_skills_selected.sort_values(by="LEVEL_VAL", ascending=False)
        # For each row, add the USER_ID to the dictionary for the corresponding skill
        for _, skill_row in df_query_skills_sorted.iterrows():
            # Directly use the skill as key and append the USER_ID
            skill_to_user_ids[current_tool].append(skill_row["User_ID"])
    
    return skill_to_user_ids