import pandas as pd
from collections import defaultdict

def compute_ranking(responsibility_top_candidates, skill_to_user_ids, resp_weight, skill_weight):
    """
    Calculate the global ranking of candidates based on responsibilities and skills.

    Args:
        responsibility_top_candidates (dict): Dictionary with responsibilities as keys and lists of candidate IDs as values.
        skill_to_user_ids (dict): Dictionary with skills/tools as keys and lists of candidate IDs as values.
        resp_weight (float): Weight for responsibility scores in the global ranking (default: 0.5).
        skill_weight (float): Weight for skill scores in the global ranking (default: 0.5).

    Returns:
        pd.DataFrame: A DataFrame containing the ranked candidates with their global scores.
    """
    # Fixed weights for count vs position components
    count_weight = 0.1
    position_weight = 0.9
    
    # Validate weights
    if resp_weight + skill_weight != 1.0:
        print(f"Warning: resp_weight ({resp_weight}) + skill_weight ({skill_weight}) != 1.0")
        # Normalize weights
        total = resp_weight + skill_weight
        resp_weight = resp_weight / total
        skill_weight = skill_weight / total
        print(f"Normalized weights: resp_weight={resp_weight}, skill_weight={skill_weight}")
    
    # 1. Calculate scores for responsibilities
    # a. Simple count
    resp_count = defaultdict(int)
    for resp, candidates in responsibility_top_candidates.items():
        for cand in candidates:
            resp_count[cand] += 1

    # b. Position-based score
    resp_pos_score = defaultdict(float)
    for resp, candidates in responsibility_top_candidates.items():
        n = len(candidates)
        for idx, cand in enumerate(candidates):
            score = (n - idx) / n  # Best score = 1 for the first candidate, 1/n for the last
            resp_pos_score[cand] += score

    # c. Final responsibility score using provided weights
    final_resp_score = {}
    for cand in resp_count:
        final_resp_score[cand] = count_weight * resp_count[cand] + position_weight * resp_pos_score[cand]

    # 2. Calculate scores for skills/tools
    # a. Simple count
    skills_count = defaultdict(int)
    for skill, candidates in skill_to_user_ids.items():
        for cand in candidates:
            skills_count[cand] += 1

    # b. Position-based score
    skills_pos_score = defaultdict(float)
    for skill, candidates in skill_to_user_ids.items():
        n = len(candidates)
        for idx, cand in enumerate(candidates):
            score = (n - idx) / n
            skills_pos_score[cand] += score

    # c. Final skills score using provided weights
    final_skills_score = {}
    for cand in skills_count:
        final_skills_score[cand] = count_weight * skills_count[cand] + position_weight * skills_pos_score[cand]

    # 3. Use union of candidates with intelligent weighting
    all_candidates = set(final_resp_score.keys()) | set(final_skills_score.keys())

    # 4. Calculate global weighted score using provided resp_weight and skill_weight
    global_scores = {}
    for cand in all_candidates:
        # Check if candidate exists in both dimensions
        has_resp = cand in final_resp_score
        has_skills = cand in final_skills_score
        
        # Get scores (with defaults of 0)
        resp_score = final_resp_score.get(cand, 0)
        skill_score = final_skills_score.get(cand, 0)
        
        if has_resp and has_skills:
            # Candidate in both dimensions - normal weighting
            global_scores[cand] = resp_weight * resp_score + skill_weight * skill_score
        elif has_resp:
            # Candidate only in responsibilities - apply small penalty
            global_scores[cand] = resp_weight * resp_score * 0.90
        elif has_skills:
            # Candidate only in skills - apply larger penalty
            global_scores[cand] = skill_weight * skill_score * 0.5

    # 5. Create a ranked DataFrame
    df_ranked = pd.DataFrame(list(global_scores.items()), columns=["USER_ID", "SCORE_SKILLS_MISSIONS"])
    df_ranked = df_ranked.sort_values(by="SCORE_SKILLS_MISSIONS", ascending=False).reset_index(drop=True)

    return df_ranked