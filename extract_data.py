#Importer les librairies nécessaires
import os
from docling.document_converter import DocumentConverter

def convert_pdf_to_markdown(file_name):
    """
    Extraire les données d'un fichier PDF et les convertir en JSON.
    
    Args:
        file_name (str): Le nom du fichier PDF à convertir.
    
    Returns:
        markdown_output (str): Le markdown résultant contenant les données extraites.
    """
    # Initializiser le DocumentConverter
    converter = DocumentConverter()

    # Trouver 'ressources' directory relative au fichier actuel
    current_directory = os.path.dirname(__file__)
    ressources_directory = os.path.join(current_directory, "ressources")

    # Fonctionne pour trouver le chemin du fichier PDF dynamiquement
    def find_file_path(directory, file_name):
        """
        Search for a file in the given directory and return its full path.
        """
        for root, dirs, files in os.walk(directory):
            if file_name in files:
                return os.path.join(root, file_name)
        raise FileNotFoundError(f"File '{file_name}' not found in directory '{directory}'.")

    # file path trouvé dynamiquement
    pdf_file_path = find_file_path(ressources_directory, file_name)

    # convertir le fichier PDF en document
    result = converter.convert(pdf_file_path)

    # Extraire le document du résultat
    document = result.document

    # Exporter le document au format JSON
    markdown_output = document.export_to_markdown()

    return markdown_output
