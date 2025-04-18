import pandas as pd
import os

def read_csv_files(file_names):
    """
    Lit plusieurs fichiers Excel et renvoie leurs DataFrames.

    Args:
        file_names (list): Une liste de 5 noms de fichiers à lire.

    Returns:
        tuple: Un tuple contenant 5 DataFrames pandas.
    """
    # Récupère le répertoire du fichier courant
    current_directory = os.path.dirname(__file__)
    # Construit le chemin vers le répertoire "ressources" relatif au fichier courant
    ressources_directory = os.path.join(current_directory, "ressources")

    # Fonction pour trouver dynamiquement le chemin complet d'un fichier dans un répertoire donné
    def find_file_path(directory, file_name):
        """
        Recherche un fichier dans le répertoire donné et renvoie son chemin complet.
        """
        for root, dirs, files in os.walk(directory):
            if file_name in files:
                return os.path.join(root, file_name)
        raise FileNotFoundError(f"Fichier '{file_name}' introuvable dans le répertoire '{directory}'.")

    # Trouve dynamiquement les chemins des fichiers
    file_paths = [find_file_path(ressources_directory, file_name) for file_name in file_names]

    # Lit les fichiers Excel dans des DataFrames
    dataframes = [pd.read_csv(file_path) for file_path in file_paths]

    return tuple(dataframes)  # Renvoie les DataFrames sous forme de tuple