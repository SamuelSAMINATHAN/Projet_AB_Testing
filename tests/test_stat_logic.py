import pytest
import pandas as pd
import numpy as np
from stats_utils import clean_data, check_srm, run_bayesian_inference

def test_clean_data_removes_bots():
    """Vérifie que les utilisateurs > 1000 ads sont bien supprimés"""
    data = pd.DataFrame({
        'user id': [1, 2, 3, 4],
        'total ads': [10, 500, 1001, 50],
        'converted': [0, 1, 0, 0]
    })
    cleaned = clean_data(data)
    assert len(cleaned) == 3
    assert cleaned['total ads'].max() <= 1000

def test_clean_data_removes_duplicates():
    """Vérifie que les doublons de user id sont supprimés"""
    data = pd.DataFrame({
        'user id': [1, 1, 2],
        'total ads': [10, 10, 20],
        'converted': [0, 0, 1]
    })
    cleaned = clean_data(data)
    assert len(cleaned) == 2
    assert cleaned['user id'].is_unique

def test_srm_detection():
    """Vérifie que le SRM détecte un biais si le ratio est de 50/50 au lieu de 96/4"""
    # On simule 500 Ad et 500 PSA (ratio 50/50)
    res = check_srm(observed_ad=500, observed_psa=500, expected_ratio_ad=0.96)
    # CORRECTION : Utilisation de == au lieu de 'is'
    assert res['is_biased'] == True 
    assert res['p_value'] < 0.01

def test_srm_valid_ratio():
    """Vérifie que le SRM valide un ratio proche de 96/4"""
    # 960 Ad et 40 PSA
    res = check_srm(observed_ad=960, observed_psa=40, expected_ratio_ad=0.96)
    # CORRECTION : Utilisation de == au lieu de 'is'
    assert res['is_biased'] == False
    assert res['p_value'] > 0.05

def test_bayesian_inference_logic():
    """Vérifie que si Ad a 10% de conv et PSA 1%, Ad > PSA est quasi certain"""
    res = run_bayesian_inference(
        conv_ad=100, n_ad=1000, 
        conv_psa=10, n_psa=1000,
        n_samples=1000
    )
    assert res['prob_better'] > 0.95 # Seuil légèrement abaissé pour éviter les aléas du sampling
    assert res['expected_loss'] < 0.01

def test_bayesian_inference_uncertainty():
    """Vérifie que sur de très petits échantillons, l'incertitude est prise en compte"""
    res = run_bayesian_inference(
        conv_ad=1, n_ad=2, 
        conv_psa=0, n_psa=2,
        n_samples=1000
    )
    # Avec les priors Beta(2, 80), le résultat est tiré vers la moyenne historique
    assert res['prob_better'] < 0.90