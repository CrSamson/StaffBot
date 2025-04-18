import chromadb
import uuid

def compute_mission_to_user_ids(df_completed_user, json_res, top_percentage=0.3):
    """
    Calculate the top candidates for each responsibility based on mission similarities.
    
    Args:
        df_completed_user (pd.DataFrame): DataFrame containing user missions and embeddings.
        json_res (dict): JSON object containing job responsibilities.
        top_percentage (float): Percentage of top candidates to select for each responsibility.
        
    Returns:
        dict: A dictionary where keys are responsibilities and values are lists of top candidate User_IDs.
    """
    # Créer un client en mémoire sans persistance sur disque
    client = chromadb.Client()
    
    # Créer une collection pour stocker les missions
    collection_mission = client.get_or_create_collection(name="missions_candidates")
    
    # Vérifier si la collection est vide avant d'ajouter les missions
    if collection_mission.count() == 0:
        # Ajouter les missions des utilisateurs à la collection
        print("Ajout des missions à la base vectorielle en mémoire...")
        for _, row in df_completed_user.iterrows():
            user_id = row["USER_ID"]
            mission_sentences = row["MISSIONS_lemmatized"]
            mission_embeddings = row["MISSIONS_embeddings"]
            for sentence, embedding in zip(mission_sentences, mission_embeddings):
                # Convert embedding to a list if necessary
                embedding_list = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
                collection_mission.add(
                    documents=[sentence],
                    embeddings=[embedding_list],
                    metadatas={"User_ID": user_id},
                    ids=[str(uuid.uuid4())]
                )
    else:
        print(f"Utilisation de la collection existante avec {collection_mission.count()} missions.")
    
    # Extract responsibilities from the JSON
    responsibilities = json_res["responsibilities"]
    
    # Dictionary to store top candidates for each responsibility
    responsibility_top_candidates = {}
    
    # Process each responsibility
    for resp in responsibilities:
        # Query the collection for the current responsibility
        query_result = collection_mission.query(
            query_texts=[resp],
            n_results=100,  # Adjust based on collection size
            include=["distances", "metadatas"]
        )
        
        # Track the best distance for each candidate
        candidate_best_distance = {}
        for i, metadata in enumerate(query_result["metadatas"][0]):
            user_id = metadata["User_ID"]
            distance = query_result["distances"][0][i]
            if user_id in candidate_best_distance:
                candidate_best_distance[user_id] = min(candidate_best_distance[user_id], distance)
            else:
                candidate_best_distance[user_id] = distance
        
        # Sort candidates by distance
        sorted_candidates = sorted(candidate_best_distance.items(), key=lambda x: x[1])
        
        # Calculate the number of top candidates to select
        num_candidates = len(sorted_candidates)
        top_n = max(1, int(num_candidates * top_percentage))
        
        # Get the User_IDs of the top candidates
        top_candidate_ids = [user_id for user_id, _ in sorted_candidates[:top_n]]
        
        # Store the result in the dictionary
        responsibility_top_candidates[resp] = top_candidate_ids
    
    return responsibility_top_candidates