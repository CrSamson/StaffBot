import pandas as pd
from extract_data import convert_pdf_to_markdown
from llm import extract_json_from_markdown
from csv_to_dataframe import read_csv_files
from compute_disponilite import calculate_availability_score
from clean_dataframes import clean_dataframes
from semantic_search_skills import compute_skill_to_user_ids
from preprocess_missions import preprocess_missions
from semantic_search_missions import compute_mission_to_user_ids
from ranking_skills_missions import compute_ranking
from language_score import calculate_language_score
from compute_global_score import calculate_global_score
import re
import gc  # Pour le garbage collector
import os  # Pour redémarrer le processus
import streamlit as st

def main(poste):
    """
    Fonction principale orchestrant le processus de matching des candidats avec le poste.
    
    Args:
        poste (str): Nom du poste à traiter (ex: "Data Analyst", "Data Engineer", "Scrum")
    
    Returns:
        pd.DataFrame: DataFrame contenant les 5 meilleurs candidats
    """
    st.write(f"Analyse du poste: **{poste}**")
    
    # Afficher une barre de progression
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # 1. Conversion du PDF du poste en markdown
    status_text.write("Étape 1/12: Conversion du PDF en texte...")
    markdown_output = convert_pdf_to_markdown(f"{poste}.pdf")
    progress_bar.progress(1/12)
    
    # 2. Extraction des données structurées (JSON) à partir du markdown
    status_text.write("Étape 2/12: Extraction des données structurées...")
    json_res = extract_json_from_markdown(markdown_output)
    progress_bar.progress(2/12)
    
    # 2.1 Regex pour déterminer la catégorie du poste (technique ou gestion)
    pattern_technique = r'\b(?:\w*\s*)?techn?iqu[e]s?\b'
    plus_technique = bool(re.search(pattern_technique, json_res["catégorie_poste"], re.IGNORECASE))
    pattern_gestion = r'\b(?:\w*\s*)?gestion\b'
    plus_gestion = bool(re.search(pattern_gestion, json_res["catégorie_poste"], re.IGNORECASE))
    
    if plus_gestion:
        st.info("Poste orienté gestion détecté - Priorisation des missions")
    else:
        st.info("Poste orienté technique détecté - Priorisation des compétences")
    
    # 3. Lecture des fichiers CSV
    status_text.write("Étape 3/12: Lecture des fichiers de données...")
    file_names = [
        "HCK_HEC_LANG.csv",
        "HCK_HEC_SKILLS.csv",
        "HCK_HEC_STAFFING.csv",
        "HCK_HEC_USER.csv",
        "HCK_HEC_XP.csv"
    ]
    lang_df, skills_df, staffing_df, user_df, xp_df = read_csv_files(file_names)
    progress_bar.progress(3/12)
    
    # 4. Nettoyage des DataFrames
    status_text.write("Étape 4/12: Nettoyage des données...")
    df_clean = clean_dataframes(user_df, xp_df, skills_df, lang_df, staffing_df)
    progress_bar.progress(4/12)
    
    # 5-7. Flux adapté selon le type de poste
    if plus_gestion:
        # 5a. Prétraitement des missions
        status_text.write("Étape 5/12: Prétraitement des missions...")
        df_postprocess = preprocess_missions(df_clean)
        progress_bar.progress(5/12)
        
        # 6a. Recherche par missions d'abord
        status_text.write("Étape 6/12: Recherche de candidats par missions...")
        responsibility_top_candidates = compute_mission_to_user_ids(df_postprocess, json_res)
        progress_bar.progress(6/12)
        
        # Extraire candidats qualifiés par missions
        top_candidates_per_resp = [users[:5] for users in responsibility_top_candidates.values()]
        mission_qualified_users = list(set([user_id for user_list in top_candidates_per_resp for user_id in user_list]))
        st.write(f"Nombre de candidats qualifiés par missions: {len(mission_qualified_users)}")
        
        # Filtrer le dataframe skills
        filtered_skills_df = skills_df[skills_df['USER_ID'].isin(mission_qualified_users)]
        
        # 7a. Recherche des compétences parmi candidats qualifiés
        status_text.write("Étape 7/12: Recherche des compétences...")
        skill_to_user_ids = compute_skill_to_user_ids(filtered_skills_df, json_res)
        progress_bar.progress(7/12)
        
    else:
        # 5b. Recherche par compétences d'abord
        status_text.write("Étape 5/12: Recherche de candidats par compétences...")
        skill_to_user_ids = compute_skill_to_user_ids(skills_df, json_res)
        progress_bar.progress(5/12)
    
        # Extraire la liste des candidats qualifiés par compétences
        skill_qualified_users = list(set([user_id for users in skill_to_user_ids.values() for user_id in users]))
        st.write(f"Nombre de candidats qualifiés par compétences: {len(skill_qualified_users)}")
    
        # 6b. Prétraitement des missions
        status_text.write("Étape 6/12: Prétraitement des missions...")
        df_postprocess = preprocess_missions(df_clean)
        progress_bar.progress(6/12)
    
        # Filtrer pour les candidats qualifiés par compétences
        filtered_df_postprocess = df_postprocess[df_postprocess['USER_ID'].isin(skill_qualified_users)]
    
        # 7b. Recherche des missions
        status_text.write("Étape 7/12: Recherche des candidats par missions...")
        responsibility_top_candidates = compute_mission_to_user_ids(filtered_df_postprocess, json_res)
        progress_bar.progress(7/12)
    
    # 8. Re-ranking avec pondération différente selon le type de poste
    status_text.write("Étape 8/12: Calcul des scores skills/missions...")
    if plus_gestion:
        # Pour gestion: missions plus importantes
        df_ranking_initial = compute_ranking(responsibility_top_candidates, skill_to_user_ids, 
                                           resp_weight=0.9, skill_weight=0.1)
    else:
        # Pour technique: compétences plus importantes
        df_ranking_initial = compute_ranking(responsibility_top_candidates, skill_to_user_ids,
                                           resp_weight=0.1, skill_weight=0.9)
    progress_bar.progress(8/12)
    
    # 9. Fusion des DataFrames
    status_text.write("Étape 9/12: Fusion des données...")
    df_final = pd.merge(df_ranking_initial[['USER_ID', 'SCORE_SKILLS_MISSIONS']],
                        df_postprocess,
                        on='USER_ID',
                        how='inner')
    progress_bar.progress(9/12)
    
    # 10. Calcul du score de disponibilité
    status_text.write("Étape 10/12: Calcul des scores de disponibilité...")
    df_score_dispo = calculate_availability_score(df_final, json_res["mandate_duration"])
    progress_bar.progress(10/12)
    
    # 11. Calcul du score de langue
    status_text.write("Étape 11/12: Calcul des scores de langue...")
    df_score_language = calculate_language_score(df_score_dispo, json_res["languages"])
    progress_bar.progress(11/12)
    
    # 12. Calcul du score global avec pondération adaptée
    status_text.write("Étape 12/12: Calcul du score global...")
    if plus_gestion:
        weights = {"skills": 0.6, "dispo": 0.25, "language": 0.15}
    else:
        weights = {"skills": 0.8, "dispo": 0.10, "language": 0.10}
        
    df_candidats = calculate_global_score(df_score_language, weights)
    df_top5_candidats = df_candidats.nlargest(5, 'SCORE_GLOBALE')
    df_top5_candidats = df_top5_candidats[["USER_ID", "SCORE_SKILLS_MISSIONS", "SCORE_DISPO", "SCORE_LANGUAGE", "SCORE_GLOBALE"]]
    progress_bar.progress(12/12)
    
    # Affichage et sauvegarde
    status_text.write("✅ Traitement terminé!")
    df_top5_candidats.to_csv(f"{poste}_top5_candidats.csv", index=False)
    
    # Nettoyer la mémoire
    gc.collect()
    
    return df_top5_candidats

# Interface Streamlit
def streamlit_ui():
    st.title("RoboStaff - Système de Matching de Candidats")
    st.markdown("### Sélectionnez un poste à analyser")
    
    # Liste des postes disponibles
    postes_disponibles = ["Data Analyst", "Data Engineer", "Scrum"]
    
    # Sélection du poste via une liste déroulante
    poste_selectionne = st.selectbox("Poste :", postes_disponibles)
    
    # Bouton d'exécution
    if st.button("Lancer l'analyse"):
        try:
            with st.spinner(f"Analyse en cours pour le poste {poste_selectionne}..."):
                # Exécuter la fonction principale avec le poste sélectionné
                top_candidats = main(poste_selectionne)
                
                # Afficher les résultats
                st.success(f"Analyse terminée pour le poste {poste_selectionne}!")
                st.markdown("### Top 5 des candidats")
                st.dataframe(top_candidats)
                
                # Proposer de télécharger le fichier CSV
                csv_path = f"{poste_selectionne}_top5_candidats.csv"
                with open(csv_path, "rb") as file:
                    st.download_button(
                        label="Télécharger le CSV",
                        data=file,
                        file_name=csv_path,
                        mime="text/csv"
                    )
        except Exception as e:
            st.error(f"Une erreur s'est produite: {str(e)}")
            
if __name__ == "__main__":
    streamlit_ui()