import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from stats_utils import (
    clean_data, check_srm, run_bayesian_inference, 
    calculate_incremental_profit, get_saturation_stats
)

# Configuration de la page
st.set_page_config(
    page_title="Marketing Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Correction du contraste Dark/Light Mode via CSS
st.markdown("""
    <style>
    /* Conteneur principal */
    .main { background-color: transparent; }
    
    /* Style des Metrics pour assurer la visibilité du texte quel que soit le mode */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
    }
    
    /* Boites d'information personnalisées */
    .metric-card {
        background-color: rgba(128, 128, 128, 0.1);
        border: 1px solid rgba(128, 128, 128, 0.2);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    
    /* Fix pour les textes blancs sur fond clair en mode auto */
    .stMarkdown, .stText {
        color: inherit;
    }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configuration")
    uploaded_file = st.file_uploader("Dataset marketing_AB.csv", type="csv")
    
    st.divider()
    st.header("💰 Variables Économiques")
    margin = st.number_input("Marge par Conversion ($)", value=40.0, step=1.0, help="Profit net généré par une vente (hors frais marketing).")
    
    # Utilisation du CPM pour plus de clarté métier
    cpm = st.slider("CPM (Coût pour 1000 impressions)", 0.5, 20.0, 5.0, format="%.2f$")
    ad_cost_unit = cpm / 1000
    
    sim_volume = st.number_input("Audience cible (Volume)", value=1_000_000, step=100000, help="Nombre d'utilisateurs sur lesquels simuler le déploiement.")

# --- LOGIQUE PRINCIPALE ---
if uploaded_file:
    df_raw = pd.read_csv(uploaded_file)
    df = clean_data(df_raw)
    
    st.title("🛡️ Decision Support System")
    st.caption("Analyse de performance incrémentale et validation statistique du ROI.")

    # Calculs préalables
    counts = df['test group'].value_counts()
    srm = check_srm(counts.get('ad', 0), counts.get('psa', 0))
    
    tab1, tab2, tab3 = st.tabs(["🛡️ Validité & Diagnostic", "📈 Saturation Publicitaire", "💵 Modélisation du Profit"])

    # --- TAB 1 : DIAGNOSTIC ---
    with tab1:
        st.subheader("Intégrité du Test A/B")
        st.info("**Pourquoi cette étape ?** On vérifie que la répartition entre le groupe Ad (Publicité) et PSA (Témoin) est naturelle. Si le ratio est anormal (SRM), les résultats sont biaisés.")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if srm['is_biased']:
                st.error(f"⚠️ Alerte SRM : Biais détecté (p={srm['p_value']:.4f})")
                st.markdown("Le déséquilibre entre les groupes est trop fort pour être dû au hasard.")
            else:
                st.success(f"✅ Randomisation fiable (p={srm['p_value']:.4f})")
            
            fig_pie = px.pie(
                values=[counts.get('ad', 0), counts.get('psa', 0)], 
                names=['Groupe Ad', 'Groupe PSA'], 
                hole=0.4,
                color_discrete_sequence=['#1f77b4', '#7f7f7f'],
                title="Répartition de l'échantillon"
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            summary = df.groupby('test group')['converted'].mean().reset_index()
            fig_bar = px.bar(
                summary, x='test group', y='converted', 
                color='test group',
                title="Taux de Conversion Brut par Groupe",
                labels={'converted': 'Taux de Conv.', 'test group': 'Groupe'},
                color_discrete_map={'ad': '#1f77b4', 'psa': '#7f7f7f'}
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    # --- TAB 2 : SATURATION ---
    with tab2:
        st.subheader("Analyse de la Fréquence d'Exposition")
        st.markdown("""
        **L'objectif :** Identifier le 'Frequency Cap'. C'est le point où augmenter le nombre de publicités vues par personne ne génère plus de hausse significative des conversions.
        """)
        
        sat_stats = get_saturation_stats(df)
        
        fig_sat = go.Figure()
        # Ligne de tendance
        fig_sat.add_trace(go.Scatter(
            x=sat_stats.index, y=sat_stats['mean'], 
            name='Taux de Conv.', line=dict(color='#1f77b4', width=3)
        ))
        # Intervalle de confiance
        fig_sat.add_trace(go.Scatter(
            x=sat_stats.index, y=sat_stats['ci_upper'], 
            fill=None, mode='lines', line_color='rgba(31, 119, 180, 0.1)', showlegend=False
        ))
        fig_sat.add_trace(go.Scatter(
            x=sat_stats.index, y=sat_stats['ci_lower'], 
            fill='tonexty', mode='lines', line_color='rgba(31, 119, 180, 0.1)', name='Confiance (95%)'
        ))
        
        fig_sat.update_layout(
            template="plotly_white",
            xaxis_title="Nombre d'expositions (Bins)",
            yaxis_title="Taux de Conversion",
            hovermode="x unified"
        )
        st.plotly_chart(fig_sat, use_container_width=True)
        st.warning("💡 Si la courbe plafonne ou redescend, limiter l'exposition permet d'économiser du budget sans perdre de ventes.")

    # --- TAB 3 : ROI ---
    with tab3:
        st.subheader("Simulation Économique (Inférence Bayésienne)")
        
        summary_stats = df.groupby('test group')['converted'].agg(['sum', 'count'])
        bayesian = run_bayesian_inference(
            summary_stats.loc['ad', 'sum'], summary_stats.loc['ad', 'count'],
            summary_stats.loc['psa', 'sum'], summary_stats.loc['psa', 'count']
        )
        
        avg_ads = df[df['test group'] == 'ad']['total ads'].mean()
        profit_samples = calculate_incremental_profit(
            bayesian['samples_ad'], bayesian['samples_psa'],
            sim_volume, avg_ads, ad_cost_unit, margin
        )
        
        # Metrics - Utilisation de colonnes standards pour compatibilité thème
        m1, m2, m3 = st.columns(3)
        m1.metric("Probabilité Ad > PSA", f"{bayesian['prob_better']:.2%}", help="Certitude que la pub est meilleure que rien du tout.")
        m2.metric("Profit Net Moyen", f"${profit_samples.mean():,.0f}", delta=f"{profit_samples.mean():,.0f}")
        m3.metric("Risque de Perte", f"{(profit_samples < 0).mean():.2%}", delta_color="inverse")
        
        # Distribution
        fig_profit = px.histogram(
            profit_samples, nbins=50, 
            title="Distribution du Profit Incrémental Estimé",
            color_discrete_sequence=['#2ecc71'] if profit_samples.mean() > 0 else ['#e74c3c'],
            labels={'value': 'Profit Net ($)'}
        )
        fig_profit.add_vline(x=0, line_dash="dash", line_color="#34495e")
        st.plotly_chart(fig_profit, use_container_width=True)

        # Verdict Final
        st.divider()
        if profit_samples.mean() > 0 and (profit_samples < 0).mean() < 0.05:
            st.success(f"### 🚀 RECOMMANDATION : DÉPLOIEMENT\nLe gain incrémental moyen est de **${profit_samples.mean():,.0f}**. Le risque financier est jugé acceptable (<5%).")
        elif profit_samples.mean() > 0:
            st.warning(f"### ⚠️ RECOMMANDATION : DÉPLOIEMENT PRUDENT\nLe profit moyen est positif (**${profit_samples.mean():,.0f}**), mais le risque de perte est de **{(profit_samples < 0).mean():.1%}**. Surveillez les coûts.")
        else:
            st.error(f"### 🛑 RECOMMANDATION : ARRÊT\nLe coût d'acquisition dépasse la marge générée. Perte estimée : **${profit_samples.mean():,.0f}**.")

else:
    st.info("👋 Veuillez charger le fichier CSV pour activer l'analyse.")