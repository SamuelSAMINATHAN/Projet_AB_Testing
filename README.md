# Analyse d'A/B Testing Marketing : Efficacité des Campagnes Publicitaires

## Objectif du projet
L'objectif de ce projet est de valider scientifiquement l'impact d'une campagne de marketing digital sur le comportement d'achat. Au lieu de se fier à des métriques de surface, nous utilisons une approche rigoureuse pour isoler l'**effet causal** de la publicité et identifier les leviers d'**optimisation du ROI** via le "Dayparting".

**Stack Technique :** Python (Pandas, Scipy, Statsmodels), Streamlit (Dashboard), Pytest.

---

## 1. Intégrité des Données et Sanity Checks
Avant toute analyse, nous validons la qualité de l'échantillon pour éviter les biais de sélection.

### Test de Sample Ratio Mismatch (SRM)
Nous vérifions que la répartition **96/4** entre le groupe Ad (Exposé) et PSA (Contrôle) est statistiquement conforme aux attentes du plan d'expérience.
![Taille de l'échantillon](reports/Taille%20de%20l’échantillon.png)

### Traitement des Outliers
L'analyse du volume publicitaire a révélé des valeurs extrêmes. Un nettoyage basé sur le 99ème percentile a été appliqué pour stabiliser les moyennes.

| Avant suppression | Après suppression |
|---|---|
| ![Total publicités avant](reports/Total%20de%20publicités.png) | ![Total publicités après](reports/Total%20de%20publicités%20vues%20par%20utilisateur%20(APRÈS%20suppression).png) |

---

## 2. Analyse de l'Efficacité (A/B Testing)
L'évaluation de la performance repose sur une double validation : fréquentiste et bayésienne.

* **Approche Fréquentiste :** Le test du Chi-deux confirme une différence significativement haute du taux de conversion pour le groupe exposé.
* **Approche Bayésienne :** L'utilisation de distributions Beta montre une probabilité proche de **100%** que la variante "Ad" surperforme le "PSA".

![Taux de conversion](reports/Taux%20de%20conversion.png)

---

## 3. Modélisation de la Causalité
Pour isoler l'effet propre à la publicité des variables de confusion (jour de la semaine, heure de la journée), nous avons implémenté une **Régression Logistique**.

**Résultats clés :**
* **Odds Ratios :** L'exposition publicitaire augmente significativement les chances de conversion, indépendamment du timing.
* **Densité d'exposition :** Analyse de la répartition des impressions par heure pour détecter les pics d'engagement.

![Densité exposition](reports/Densité%20de%20l’exposition%20publicitaire%20par%20heure.png)

---

## 4. Optimisation Temporelle (Dayparting)
L'analyse matricielle identifie les créneaux stratégiques ("Golden Hours") où le différentiel de conversion est le plus élevé pour le groupe Ad, permettant une allocation budgétaire intelligente.

![Analyse des heures stratégiques](reports/Analyse%20des%20heures%20stratégiques.png)

---

## 5. Structure du Projet
Le projet est organisé autour de trois composants clés :
-   **app.py** : Dashboard interactif (Streamlit) pour visualiser la saturation et les simulations de profit.
-   **stats_utils.py** : Moteur logique contenant les calculs statistiques (SRM, Inférence Bayésienne, ROI).
-   **test_stat_logic.py** : Suite de tests unitaires (Pytest) pour valider la robustesse des algorithmes.

---

## Recommandations Stratégiques
1.  **Validation du Lift :** La publicité génère un lift incrémental réel et statistiquement validé ; le maintien du budget est recommandé.
2.  **Optimisation "Hour-Day" :** Prioriser les investissements sur les créneaux identifiés dans la Heatmap (ex: Samedi matin) où l'engagement est à son maximum.
3.  **Gestion de la Saturation :** Utiliser le dashboard pour surveiller le volume de publicités par utilisateur et éviter la fatigue publicitaire au-delà du 99ème percentile.

---

## Installation et Utilisation
Le projet utilise `uv` pour la gestion des dépendances :
1.  Installer les dépendances : `uv sync`
2.  Lancer le dashboard : `uv run streamlit run app.py`
3.  Exécuter les tests : `uv run pytest tests/`