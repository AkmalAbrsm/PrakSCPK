import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="🌍 Country Economy Index Analyzer", layout="wide")

def validity_check(matrix, weights):
    n = len(matrix)
    RI = {2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32,
          8: 1.41, 9: 1.45, 10: 1.51, 11: 1.53, 12: 1.54, 13: 1.56, 14: 1.57}
    lamda_max = np.mean(matrix @ weights / weights)
    CI = (lamda_max - n) / (n - 1)
    CR = CI / RI.get(n, 1.12)

    st.markdown("### ✅ Consistency Check")
    st.write(f"λ max: `{lamda_max:.4f}`")
    st.write(f"CI (Consistency Index): `{CI:.3f}`")
    st.write(f"CR (Consistency Ratio): `{CR:.3f}`")

    if CR <= 0.1:
        st.success("Matrix is **consistent** ✅ (CR ≤ 0.1)")
    else:
        st.error("Matrix is **inconsistent** ❌ (CR > 0.1). Please review your weights.")

st.title("📊 Country Economy Index Analysis using AHP")
st.write("##### Elyuzar Fazlurrahman - 123230216")
st.write("##### Akmal Abrisam - 123230084")

df = pd.read_csv("download.csv")
df_clean = df[['Rank', 'Country', 'Cost of Living Index', 'Rent Index',
               'Groceries Index', 'Restaurant Price Index', 'Local Purchasing Power Index']]

tab1, tab2 = st.tabs(["📊 Data & Explore", "⚖️ AHP Analysis"])

with tab1:
    st.subheader("📄 Cleaned Country Index Dataset")
    st.dataframe(df_clean, use_container_width=True)

    st.subheader("🔍 Explore Top 10 Countries by Specific Index")
    category = st.selectbox(
        "Select a category:",
        (
            'Cost of Living Index',
            'Rent Index',
            'Groceries Index',
            'Restaurant Price Index',
            'Local Purchasing Power Index'
        )
    )
    option = st.radio("Show:", ['Highest', 'Lowest'], horizontal=True)

    if st.button("🎯 Show Top 10"):
        if option == 'Highest':
            top10 = df_clean.nlargest(10, category)
            st.subheader(f"📈 Top 10 Countries with Highest {category}")
        else:
            top10 = df_clean.nsmallest(10, category)
            st.subheader(f"📉 Top 10 Countries with Lowest {category}")
        st.dataframe(top10[['Rank', 'Country', category]], use_container_width=True)

with tab2:
    col_index_names = [
        'Cost of Living Index',
        'Rent Index',
        'Groceries Index',
        'Restaurant Price Index',
        'Local Purchasing Power Index'
    ]

    normalization_direction = {
        'Cost of Living Index': 'Lower is better',
        'Rent Index': 'Lower is better',
        'Groceries Index': 'Lower is better',
        'Restaurant Price Index': 'Lower is better',
        'Local Purchasing Power Index': 'Higher is better'
    }

    st.subheader("⚖️ Input Importance (Weights) for Each Index")
    st.markdown("Higher values indicate more importance. Values must be between 1 and 10.")

    with st.form("weight_form"):
        weight_cols = st.columns(5)
        col_index_values = np.array([
            weight_cols[i].number_input(name, min_value=1, max_value=10, value=5, step=1, key=name)
            for i, name in enumerate(col_index_names)
        ])
        submitted = st.form_submit_button("✅ Submit Weights")

    if submitted:
        st.subheader("📊 AHP Pairwise Comparison & Scoring")

        n = len(col_index_values)
        pairwise_matrix = np.array([
            [col_index_values[i] / col_index_values[j] for j in range(n)]
            for i in range(n)
        ])

        st.markdown("#### 📌 Pairwise Comparison Matrix")
        st.dataframe(pd.DataFrame(pairwise_matrix, columns=col_index_names, index=col_index_names), use_container_width=True)

        normalized_matrix = pairwise_matrix / pairwise_matrix.sum(axis=0)
        weights = normalized_matrix.mean(axis=1)

        st.markdown("#### 📈 Calculated Weights")
        for name, w in zip(col_index_names, weights):
            st.write(f"- **{name}**: `{w:.4f}`")

        validity_check(pairwise_matrix, weights)

        st.markdown("### 🧮 Computing Final Scores")
        df_ranked = df_clean.copy()

        for col in col_index_names:
            col_min = df_ranked[col].min()
            col_max = df_ranked[col].max()
            if normalization_direction[col] == "Lower is better":
                df_ranked[f"Normalized_{col}"] = (col_max - df_ranked[col]) / (col_max - col_min)
            else:
                df_ranked[f"Normalized_{col}"] = (df_ranked[col] - col_min) / (col_max - col_min)

        df_ranked["final_score"] = sum(
            df_ranked[f"Normalized_{col}"] * weights[i]
            for i, col in enumerate(col_index_names)
        )
        
        st.markdown("## 🏆 Top 10 Countries by Final Weighted Score")
        columns = ['Rank', 'Country'] + col_index_names
        top10_final = df_ranked.sort_values(by="final_score", ascending=False).head(10)
        st.dataframe(top10_final[columns + ["final_score"]], use_container_width=True)

        st.markdown("### 📊 Final Score Comparison Bar Chart (Matplotlib)")
        fig, ax = plt.subplots(figsize=(10, 6))
        top10_final_sorted = top10_final.sort_values(by='final_score')
        ax.barh(top10_final_sorted['Country'], top10_final_sorted['final_score'], color='skyblue')
        ax.set_xlabel('Final Score')
        ax.set_title('Top 10 Countries by Final Score')
        for i, score in enumerate(top10_final_sorted['final_score']):
            ax.text(score + 0.01, i, f"{score:.3f}", va='center')
        st.pyplot(fig)

        st.markdown("## 📥 Download Full Result")
        csv = df_ranked.sort_values(by="final_score", ascending=False)[columns + ["final_score"]].to_csv(index=False)
        st.download_button("📥 Download Ranked CSV", csv, file_name="ranked_countries.csv", mime="text/csv")
