import pandas as pd
import re

def calculate_language_score(df_candidats, required_languages):
    """
    Calcule le score de langue pour chaque candidat en se basant sur la colonne "languages_info".
    La fonction fonctionne quel que soit le format de la liste fournie 
    (ex. ['Français (essentiel)', 'Anglais (fonctionnel)'] ou ['French', 'English']) 
    et que les valeurs dans "languages_info" soient en anglais.
    
    Pour chaque langue requise, si elle est trouvée dans "languages_info" (après traduction en anglais si nécessaire),
    le niveau (un nombre entre 0 et 100) est extrait, converti sur une échelle de 0 à 10 (ex. 100 → 10) et
    utilisé pour calculer la moyenne. Si la langue n'est pas trouvée, le score pour cette langue est 0.
    
    Args:
        df_candidats (pd.DataFrame): DataFrame contenant la colonne "languages_info".
        required_languages (list): Liste de langues requises (ex. ['Français (essentiel)', 'Anglais (fonctionnel)'] 
                                   ou ['French', 'English']).
        
    Returns:
        pd.DataFrame: DataFrame avec une nouvelle colonne "SCORE_LANGUAGE" indiquant le score moyen sur 10.
    """
    # Créer une copie pour éviter le SettingWithCopyWarning
    df_candidats = df_candidats.copy()

    def translate_language(language):
        """
        Retourne la version traduite en anglais (en minuscules) de la langue donnée,
        si celle-ci est en français. Sinon, retourne la langue en minuscule.
        Exemple : "français" devient "french", "anglais" devient "english",
                  tandis que "english" reste "english".
        """
        translations = {
            "français": "french",
            "anglais": "english"
        }
        return translations.get(language.lower(), language.lower())

    def calculate_score(row):
        total_score = 0
        num_languages = len(required_languages)
        languages_info = row["languages_info"].strip().lower()
        
        for lang in required_languages:
            # Extraire la partie principale avant la parenthèse pour le matching
            lang_clean = re.split(r"\s*\(", lang)[0].strip().lower()
            target_language = translate_language(lang_clean)
            pattern = re.compile(
                rf"({lang_clean}|{target_language})\s*\((?P<level>\d+(\.\d+)?)\)",
                re.IGNORECASE
            )
            match = pattern.search(languages_info)
            if match:
                level = float(match.group("level"))
                language_score = level / 10
                total_score += language_score
            else:
                total_score += 0

        average_score = total_score / num_languages if num_languages > 0 else 0
        return average_score

    df_candidats.loc[:, "SCORE_LANGUAGE"] = df_candidats.apply(calculate_score, axis=1)
    # Filtrer les candidats avec un score de langue inférieur à 50% (score <= 5 sur 10)
    df_candidats = df_candidats[df_candidats["SCORE_LANGUAGE"] > 5]
    return df_candidats