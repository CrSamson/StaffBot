import pandas as pd

def calculate_global_score(df, weights):
    """
    Calcule le score global pour chaque candidat en normalisant uniquement SCORE_SKILLS_MISSIONS.
    Les colonnes attendues dans le DataFrame sont :
      - SCORE_SKILLS_MISSIONS (sur une échelle quelconque)
      - SCORE_DISPO (sur une échelle de 0 à 1)
      - SCORE_LANGUAGE (sur une échelle de 0 à 10)
    
    Args:
        df (pd.DataFrame): DataFrame contenant les scores individuels.
        weights (dict): Dictionnaire des pondérations pour chaque score,
                        par exemple {"skills": 0.7, "dispo": 0.2, "language": 0.1}.
    
    Returns:
        pd.DataFrame: DataFrame avec une colonne SCORE_GLOBALE triée par ordre décroissant.
    """
    # Créer une copie pour éviter le SettingWithCopyWarning
    df = df.copy()
    
    # Normaliser uniquement SCORE_SKILLS_MISSIONS (par quantiles)
    df.loc[:, "SCORE_SKILLS_MISSIONS"] = (df["SCORE_SKILLS_MISSIONS"] - df["SCORE_SKILLS_MISSIONS"].min()) / (df["SCORE_SKILLS_MISSIONS"].max() - df["SCORE_SKILLS_MISSIONS"].min()) * 100
    df.loc[:, "SCORE_DISPO"] = df["SCORE_DISPO"] * 100  # Convertir SCORE_DISPO en échelle 0-100
    df.loc[:, "SCORE_LANGUAGE"] = df["SCORE_LANGUAGE"] * 10  # Convertir SCORE_LANGUAGE en échelle 0-100
    
    # Calculer le score global en utilisant les pondérations fournies    
    df.loc[:, "SCORE_GLOBALE"] = (
        weights["skills"] * df["SCORE_SKILLS_MISSIONS"] +
        weights["dispo"] * (df["SCORE_DISPO"]) + 
        weights["language"] * df["SCORE_LANGUAGE"]
    )
    
    # Trier le DataFrame par SCORE_GLOBALE en ordre décroissant
    df = df.sort_values(by="SCORE_GLOBALE", ascending=False).reset_index(drop=True)
    
    return df