import re
import nltk
import spacy
from sentence_transformers import SentenceTransformer

def preprocess_missions(df, column_name="MISSIONS"):
    """
    Preprocess the MISSIONS column of a DataFrame in multiple steps:
    1. Normalize text (lowercase, remove extra spaces).
    2. Segment text into sentences.
    3. Lemmatize sentences.
    4. Convert lemmatized sentences into embeddings.

    Args:
        df (pd.DataFrame): The input DataFrame containing the MISSIONS column.
        column_name (str): The name of the column to preprocess.

    Returns:
        pd.DataFrame: The updated DataFrame with additional columns for each preprocessing step.
    """
    # Step 1: Normalize text
    def preprocess_step1(text):
        """
        Normalize text by converting to lowercase, removing extra spaces, and preserving punctuation.
        """
        if not isinstance(text, str):
            text = str(text)
        text = text.lower().strip()
        text = re.sub(r"\s+", " ", text)
        return text

    # Apply Step 1
    df[column_name] = df[column_name].apply(preprocess_step1)

    # Step 2: Segment text into sentences
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    def preprocess_step2(text):
        """
        Segment text into sentences using NLTK's sentence tokenizer.
        """
        if not isinstance(text, str):
            text = str(text)
        sentences = nltk.tokenize.sent_tokenize(text, language='french')
        return sentences

    # Apply Step 2
    df[f"{column_name}_sentences"] = df[column_name].apply(preprocess_step2)

    # Step 3: Lemmatize sentences
    try:
        nlp = spacy.load("fr_core_news_sm")
    except Exception:
        import spacy.cli
        spacy.cli.download("fr_core_news_sm")
        nlp = spacy.load("fr_core_news_sm")

    def lemmatize_sentence(sentence):
        """
        Lemmatize a single sentence using spaCy.
        """
        doc = nlp(sentence)
        lemmatized_tokens = [token.lemma_ for token in doc]
        return " ".join(lemmatized_tokens)

    def preprocess_step3(sentences_list):
        """
        Lemmatize a list of sentences.
        """
        return [lemmatize_sentence(sentence) for sentence in sentences_list]

    # Apply Step 3
    df[f"{column_name}_lemmatized"] = df[f"{column_name}_sentences"].apply(preprocess_step3)

    # Step 4: Convert lemmatized sentences into embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')

    def compute_embeddings(sentences_list):
        """
        Compute embeddings for a list of lemmatized sentences using SentenceTransformer.
        """
        if not sentences_list or not isinstance(sentences_list, list):
            return []
        embeddings = model.encode(sentences_list, convert_to_tensor=False)
        return embeddings

    # Apply Step 4
    df[f"{column_name}_embeddings"] = df[f"{column_name}_lemmatized"].apply(compute_embeddings)

    return df