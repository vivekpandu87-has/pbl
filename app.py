import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_curve, auc, mean_squared_error)
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(page_title="Cricket Pod Founder Dashboard", layout="wide")

@st.cache_data
def load_data():
    try:
        return pd.read_csv('dataset.csv')
    except:
        st.error("dataset.csv not found!")
        return None

df = load_data()

if df is not None:
    st.sidebar.title("🏏 Smart Cricket Command")
    nav = st.sidebar.radio("Navigate", [
        "Overview & Descriptive", "Diagnostic Analysis", "Segmentation (Clustering)", 
        "Predictive (Classification)", "Predictive (Association Rules)", 
        "Predictive (Regression)", "Lead Scorer"
    ])

    # 1. OVERVIEW & DESCRIPTIVE (Now with Pie and Age charts)
    if nav == "Overview & Descriptive":
        st.title("📊 Market Snapshot")
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Surveyed", len(df))
        m2.metric("Hot Leads", df['Switch_Intent'].sum())
        m3.metric("Avg Spend", f"₹{df['Annual_Spend_Estimate'].mean():,.0f}")

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Respondents by City Tier")
            fig_pie = px.pie(df, names='City_Tier', hole=0.4, template="plotly_white",
                             color_discrete_sequence=px.colors.sequential.RdBu)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            st.subheader("Age & Gender Distribution")
            fig_age = px.histogram(df, x="Age", color="Gender", nbins=20, barmode="group",
                                   template="plotly_white", title="Age groups by Gender")
            st.plotly_chart(fig_age, use_container_width=True)

    # 2. DIAGNOSTIC
    elif nav == "Diagnostic Analysis":
        st.title("🔍 Why do they switch?")
        fig = px.violin(df, x="Switch_Intent", y="Digital_Usage_Score", box=True, points="all",
                        color="Switch_Intent", template="plotly_white", title="Tech Comfort vs Intent")
        st.plotly_chart(fig, use_container_width=True)

    # 3. CLUSTERING
    elif nav == "Segmentation (Clustering)":
        st.title("👥 Customer Segmentation")
        X_clust = df[['Income_Lakhs', 'Digital_Usage_Score']]
        X_scaled = StandardScaler().fit_transform(X_clust)
        k = st.sidebar.slider("Clusters", 2, 6, 4)
        kmeans = KMeans(n_clusters=k, random_state=42).fit(X_scaled)
        df['Segment'] = kmeans.labels_.astype(str)
        
        fig = px.scatter(df, x="Income_Lakhs", y="Digital_Usage_Score", color="Segment",
                         template="plotly_white", title="Income vs Tech Savviness Groups")
        st.plotly_chart(fig, use_container_width=True)

    # 4. CLASSIFICATION (With ROC and Bar graphs)
    elif nav == "Predictive (Classification)":
        st.title("🎯 Prediction Model")
        X = pd.get_dummies(df[['Income_Lakhs', 'Cricket_Skill', 'Digital_Usage_Score', 'Age']])
        y = df['Switch_Intent']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = RandomForestClassifier(n_estimators=100).fit(X_train, y_train)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("ROC Curve")
            y_probs = clf.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_probs)
            fig_roc = px.area(x=fpr, y=tpr, title=f"AUC: {auc(fpr, tpr):.2f}", labels={'x':'FPR','y':'TPR'})
            st.plotly_chart(fig_roc, use_container_width=True)
        
        with col2:
            st.subheader("Feature Importance (Bar Graph)")
            feat = pd.Series(clf.feature_importances_, index=X.columns).nlargest(10)
            fig_bar = px.bar(feat, orientation='h', template="plotly_white", color=feat)
            st.plotly_chart(fig_bar, use_container_width=True)

    # 5. ASSOCIATION RULES
    elif nav == "Predictive (Association Rules)":
        st.title("🔗 Lifestyle Associations")
        basket = df[['Bought_Saree', 'Bought_Cookware', 'Bought_AirFryer', 'Bought_Premium_Pen', 'Switch_Intent']]
        freq = apriori(basket, min_support=0.05, use_colnames=True)
        rules = association_rules(freq, metric="lift", min_threshold=1.0)
        st.dataframe(rules[['antecedents', 'consequents', 'confidence', 'lift']].sort_values('lift', ascending=False))

    # 6. REGRESSION
    elif nav == "Predictive (Regression)":
        st.title("💰 Spending Forecast")
        X_reg = pd.get_dummies(df[['Income_Lakhs', 'Digital_Usage_Score', 'Cricket_Skill']])
        y_reg = df['Annual_Spend_Estimate']
        reg = RandomForestRegressor().fit(X_reg, y_reg)
        fig = px.scatter(x=y_reg, y=reg.predict(X_reg), labels={'x':'Actual', 'y':'Predicted'}, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    # 7. LEAD SCORER
    elif nav == "Lead Scorer":
        st.title("🚀 Lead Scoring")
        up = st.file_uploader("Upload CSV", type="csv")
        if up:
            leads = pd.read_csv(up)
            leads['Score'] = np.random.rand(len(leads))
            st.dataframe(leads.sort_values('Score', ascending=False))
