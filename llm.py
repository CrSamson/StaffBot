# Librairies
from langchain_core.output_parsers import JsonOutputParser
from groq import Groq
from dotenv import load_dotenv
import os

def extract_json_from_markdown(markdown):
    """
    Traite une chaîne markdown en utilisant l'API Groq et extrait des informations structurées sous forme de JSON.

    Args:
        markdown (str): La chaîne markdown à traiter.

    Returns:
        json_res (dict): Les informations extraites au format JSON.
    """
    # Load environment variables from the .env file
    dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
    load_dotenv(dotenv_path)

    # Load the API key from the .env file
    api_key = os.getenv("GROQ_API_Token")
    if not api_key:
        print("Current Working Directory:", os.getcwd())
        raise ValueError("API key not found. Please set GROQ_API_Token in your .env file.")

    # Initialize the Groq client
    groq = Groq(api_key=api_key)

    # Define the prompt for the Groq API
    prompt = f"""
        ### TEXTE EXTRAPOLÉ DU CV:
        {markdown}
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

    # Call the Groq API to process the markdown
    chat_completion = groq.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
        temperature=0.5
    )
    content = chat_completion.choices[0].message.content

    # Fait le parsing du content en JSON
    json_parser = JsonOutputParser()
    json_res = json_parser.parse(content)

    return json_res