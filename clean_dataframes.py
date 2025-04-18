import pandas as pd

def clean_dataframes(df_user, df_xp, df_skill, df_lang, df_staffing):
    """
    Nettoie et agrège plusieurs DataFrames pour créer un DataFrame complet de candidats.

    Args:
        df_user (pd.DataFrame): DataFrame contenant les informations de base des utilisateurs.
        df_xp (pd.DataFrame): DataFrame contenant les expériences (missions) des utilisateurs.
        df_skill (pd.DataFrame): DataFrame contenant les compétences (skills) des utilisateurs.
        df_lang (pd.DataFrame): DataFrame contenant les compétences linguistiques des utilisateurs.
        df_staffing (pd.DataFrame): DataFrame contenant les informations de staffing des utilisateurs.

    Returns:
        pd.DataFrame: DataFrame complet nettoyé (completed user clean) issu de la fusion des différents DataFrames.
    """
    # 1. Suppression des doublons dans le DataFrame des expériences
    df_xp.drop_duplicates(inplace=True)
    
    # 2. Agrégation des descriptions de missions par USER_ID
    # On concatène les descriptions de mission avec un point pour séparer chaque mission
    df_xp_grouped = df_xp.groupby("USER_ID")["MISSION_DSC"].apply(lambda x: ". ".join(x)).reset_index()
    
    # 3. Renommage de la colonne agrégée en "MISSIONS"
    df_xp_grouped.rename(columns={"MISSION_DSC": "MISSIONS"}, inplace=True)
    
    # 4. Agrégation des compétences : on combine la description de la compétence et le niveau pour chaque USER_ID
    df_skill_grouped = df_skill.groupby("USER_ID").apply(
        lambda df: ', '.join(df["SKILLS_DSC"] + " (" + df["LEVEL_VAL"].astype(str) + ")")
    ).reset_index(name="skills_info")
    
    # 5. Agrégation des compétences linguistiques : on combine la description et le niveau pour chaque USER_ID
    df_lang_agg = df_lang.groupby("USER_ID").apply(
        lambda df: ', '.join(df["LANGUAGE_SKILL_DSC"] + " (" + df["LANGUAGE_SKILL_LVL"].astype(str) + ")")
    ).reset_index(name="languages_info")
    
    # 6. Fusionner tous les DataFrames agrégés avec le DataFrame de base des utilisateurs
    df_completed_user = df_user.merge(df_xp_grouped, on="USER_ID", how="left")    # Fusion avec les expériences
    df_completed_user = df_completed_user.merge(df_skill_grouped, on="USER_ID", how="left")  # Fusion avec les compétences
    df_completed_user = df_completed_user.merge(df_lang_agg, on="USER_ID", how="left")     # Fusion avec les langues
    df_completed_user = df_completed_user.merge(df_staffing, on="USER_ID", how="left")      # Fusion avec le staffing
    
    # 7. Remplacer les valeurs manquantes par des chaînes vides ou 0 pour les colonnes numériques
    df_completed_user["skills_info"] = df_completed_user["skills_info"].fillna("")
    df_completed_user["MISSIONS"] = df_completed_user["MISSIONS"].fillna("")
    df_completed_user["languages_info"] = df_completed_user["languages_info"].fillna("")
    df_completed_user["ANNEES_XP"] = df_completed_user["ANNEES_XP"].fillna(0)

    # 8. Remove candidates who are not available in any month (MONTH_1 to MONTH_12)
    month_columns = [f"MONTH_{i}" for i in range(1, 13)]  # All 12 months
    df_completed_user = df_completed_user[
        (df_completed_user[month_columns] < 100).any(axis=1)  # Keep rows where at least one month is < 100
    ]
    
    return df_completed_user