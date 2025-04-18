import pandas as pd
import re

def calculate_availability_score(df_candidats, mandate_duration_string):
    """
    Calculates an availability score for each candidate based on their availability
    for a given mandate duration, considering different time periods (days, months, years).

    Args:
        df_candidats (pd.DataFrame): DataFrame containing candidate data with month availability columns.
        mandate_duration_string (str): String containing the mandate duration (e.g., "30 jours", "10 mois", "1 year").

    Returns:
        pd.DataFrame: DataFrame with an added "SCORE_DISPO" column representing the availability score.
    """

    # Extract the duration and period from the mandate duration string
    pattern = re.compile(r"(?i)(?P<duration>\d+)\s*(?P<period>jours?|days?|mois|months?|ann[ée]e?s?|years?)")
    match = pattern.search(mandate_duration_string)

    if match:
        duration = int(match.group("duration"))
        period = match.group("period").lower()
    else:
        raise ValueError("Could not extract duration and period from mandate duration string.")

    # Convert duration to months
    if "jour" in period or "day" in period:
        duration_in_months = duration / 30  # Approximate conversion: 30 days = 1 month
    elif "année" in period or "year" in period:
        duration_in_months = duration * 12
    else:  # Assume months
        duration_in_months = duration

    # Limit duration to 12 months (maximum available in CSV)
    duration_in_months = min(duration_in_months, 12)
    num_months_to_check = int(duration_in_months)  # Ensure it's an integer for column selection

    # Define a list of month columns based on the duration
    month_columns = [f"MONTH_{i}" for i in range(1, num_months_to_check + 1)]

    # Calculate availability score for each candidate
    def calculate_score(row):
        total_score = 0
        for month_col in month_columns:
            availability = row[month_col]

            # Calculate availability score based on the given conditions
            if availability == 0:
                total_score += 1.00  # 100% available
            elif availability == 25:
                total_score += 0.75  # 75% available
            elif availability == 50:
                total_score += 0.50  # 50% available
            elif availability == 75:
                total_score += 0.25  # 25% available
            elif availability == 100:
                total_score += 0.00  # 0% available (not available)
            else:
                # Handle unexpected availability values (optional)
                total_score += (100 - availability) / 100  # Linear score

        return total_score / len(month_columns) if month_columns else 0  # Average score

    df_candidats["SCORE_DISPO"] = df_candidats.apply(calculate_score, axis=1)
    # Filter out candidates with a SCORE_DISPO below 70% (0.70)
    df_candidats = df_candidats[df_candidats["SCORE_DISPO"] >= 0.70]

    return df_candidats