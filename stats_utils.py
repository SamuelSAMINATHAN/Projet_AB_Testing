import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, Tuple

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoyage Expert : suppression des bots, gestion des doublons et intégrité des types.
    Identifie les bots basés sur un seuil physique d'exposition publicitaire (ex: > 1000 ads).
    """
    df = df.copy()
    
    # Suppression des doublons d'user id (conserve le premier)
    df = df.drop_duplicates(subset=['user id'], keep='first')
    
    # Intégrité des types
    df['converted'] = df['converted'].astype(int)
    
    # Bot Detection : Seuil d'exposition physiquement improbable (> 1000 ads)
    # Justification : Plus de 1000 pubs pour un seul utilisateur sur une période de campagne 
    # courte suggère une activité de scraping ou de bot plutôt qu'une consommation humaine réelle.
    df = df[df['total ads'] <= 1000]
    
    return df

def check_srm(observed_ad: int, observed_psa: int, expected_ratio_ad: float = 0.96) -> Dict[str, Any]:
    """
    Diagnostic d'Intégrité : Test SRM (Sample Ratio Mismatch) via Chi-deux.
    Vérifie si le split réel respecte le design expérimental de 96/4.
    """
    total = observed_ad + observed_psa
    expected_ad = total * expected_ratio_ad
    expected_psa = total * (1 - expected_ratio_ad)
    
    chi2, p_val = stats.chisquare(f_obs=[observed_ad, observed_psa], f_exp=[expected_ad, expected_psa])
    
    return {
        "p_value": p_val,
        "is_biased": p_val < 0.01,
        "observed": {"ad": observed_ad, "psa": observed_psa},
        "expected": {"ad": expected_ad, "psa": expected_psa}
    }

def run_frequentist_test(conv_ad: int, n_ad: int, conv_psa: int, n_psa: int) -> Dict[str, Any]:
    """
    Test Fréquentiste Dual : Chi-deux pour la p-value et IC du Lift.
    """
    # Contingency table
    table = [[conv_ad, n_ad - conv_ad], [conv_psa, n_psa - conv_psa]]
    chi2, p_val, _, _ = stats.chi2_contingency(table)
    
    # Lift CI (using Standard Error of Log Relative Risk approximation)
    p_ad = conv_ad / n_ad
    p_psa = conv_psa / n_psa
    lift = (p_ad - p_psa) / p_psa
    
    se_log_rr = np.sqrt((1/conv_ad) - (1/n_ad) + (1/conv_psa) - (1/n_psa))
    z = 1.96 # 95% CI
    log_rr = np.log(p_ad / p_psa)
    ci_lower = np.exp(log_rr - z * se_log_rr) - 1
    ci_upper = np.exp(log_rr + z * se_log_rr) - 1
    
    return {
        "p_value": p_val,
        "lift": lift,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper
    }

def run_bayesian_inference(conv_ad: int, n_ad: int, conv_psa: int, n_psa: int, 
                           prior_alpha: float = 2.0, prior_beta: float = 80.0, 
                           n_samples: int = 50000) -> Dict[str, Any]:
    """
    Inférence Bayésienne avec Informed Prior (Beta(2, 80) pour 2.5% conv historique).
    """
    post_ad = stats.beta(prior_alpha + conv_ad, prior_beta + n_ad - conv_ad)
    post_psa = stats.beta(prior_alpha + conv_psa, prior_beta + n_psa - conv_psa)
    
    samples_ad = post_ad.rvs(n_samples)
    samples_psa = post_psa.rvs(n_samples)
    
    prob_better = (samples_ad > samples_psa).mean()
    expected_loss = np.mean(np.maximum(0, samples_psa - samples_ad))
    
    return {
        "samples_ad": samples_ad,
        "samples_psa": samples_psa,
        "prob_better": prob_better,
        "expected_loss": expected_loss
    }

def calculate_incremental_profit(samples_ad: np.array, samples_psa: np.array, 
                                 n_sessions: int, avg_ads_per_session: float, 
                                 cost_per_ad: float, margin_per_conv: float) -> np.array:
    """
    Modélisation Économique : (Conversions Incrémentales * Marge) - (Impressions * Coût Ad).
    """
    # Conversions incrémentales simulées
    incremental_conv_rate = samples_ad - samples_psa
    incremental_conversions = incremental_conv_rate * n_sessions
    
    # Coût de la campagne Ad (Groupe Ad seulement)
    # On suppose que PSA n'a pas de coût publicitaire additionnel
    campaign_cost = n_sessions * avg_ads_per_session * cost_per_ad
    
    net_profit = (incremental_conversions * margin_per_conv) - campaign_cost
    return net_profit

def get_wilson_ci(p: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    """
    Calcule l'intervalle de confiance de Wilson pour une proportion.
    """
    if n == 0: return 0.0, 0.0
    denom = 1 + z**2/n
    center = (p + z**2/(2*n))/denom
    spread = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2))/denom
    return max(0, center - spread), min(1, center + spread)

def get_saturation_stats(df: pd.DataFrame, bins: int = 15) -> pd.DataFrame:
    """
    Analyse de Saturation : Wilson CI et volume de sessions par bin.
    """
    df = df.copy()
    df['ads_bin'] = pd.qcut(df['total ads'], q=bins, duplicates='drop')
    
    agg = df.groupby('ads_bin')['converted'].agg(['mean', 'count']).reset_index()
    cis = agg.apply(lambda row: get_wilson_ci(row['mean'], int(row['count'])), axis=1)
    
    agg['ci_lower'] = [c[0] for c in cis]
    agg['ci_upper'] = [c[1] for c in cis]
    agg['ads_bin'] = agg['ads_bin'].astype(str)
    
    return agg
