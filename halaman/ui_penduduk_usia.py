import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from model import fetch_data, train_svm_model, predict_population

@st.cache_data
def fetch_population_data():
    """Fetch population data dari Supabase dengan caching"""
    try:
        df = fetch_data(
            table_name="penduduk_usia",
            feature_columns=["id_tahun"],
            target_columns=["kategori_usia", "laki_laki", "perempuan", "total"]
        )
        if not df.empty:
            # Pastikan kolom id_tahun ada dan bertipe int
            if 'id_tahun' in df.columns:
                df['id_tahun'] = df['id_tahun'].astype(int)
            else:
                # Jika tidak ada kolom id_tahun, buat dari index atau kolom tahun
                if 'tahun' in df.columns:
                    df['id_tahun'] = df['tahun'].astype(int)
                else:
                    # Buat tahun dummy berdasarkan index
                    df['id_tahun'] = range(2020, 2020 + len(df))
            
            return df.sort_values('id_tahun')
        else:
            st.warning("Data kosong atau tidak ditemukan!")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Gagal mengambil data: {str(e)}")
        return pd.DataFrame()

def app():
    
    # Combined visualization for all age groups
    st.header("Trend Historis & Prediksi per Kelompok Umur")

    # Prepare data for visualization - group by age categories
    age_categories = ['0-14', '15-60', '60+']

    # Historical data grouped by age category
    hist_df = df.groupby(['id_tahun', 'kategori_usia']).sum().reset_index()
    hist_df = hist_df[hist_df['kategori_usia'].isin(age_categories)].tail(15)  # Last 5 years (3 categories)
    hist_df['Type'] = 'Historical'

    # Predicted data grouped by age category
    pred_viz_df = pred_df.copy()
    pred_viz_df = pred_viz_df[pred_viz_df['Kelompok Umur'].isin(age_categories)]
    pred_viz_df = pred_viz_df.rename(columns={
        'Total': 'total',
        'Laki-laki': 'laki_laki',
        'Perempuan': 'perempuan',
        'Kelompok Umur': 'kategori_usia'
    })
    pred_viz_df['Type'] = 'Predicted'
    pred_viz_df['id_tahun'] = pred_viz_df['Tahun']

    combined_df = pd.concat([hist_df, pred_viz_df])

    # Create figure
    fig = go.Figure()

    # Colors for each age category
    age_colors = {
        '0-14': '#3498db',
        '15-60': '#2ecc71',
        '60+': '#e74c3c'
    }

    # Add traces for each age category
    for age_cat in age_categories:
        # Historical data
        hist_data = combined_df[(combined_df['kategori_usia'] == age_cat) & (combined_df['Type'] == 'Historical')]
        if not hist_data.empty:
            fig.add_trace(go.Scatter(
                x=hist_data['id_tahun'],
                y=hist_data['total'],
                mode='lines+markers',
                name=f'{age_cat} (Historical)',
                line=dict(color=age_colors[age_cat], width=3),
                marker=dict(size=8)
            ))
        
        # Predicted data
        pred_data = combined_df[(combined_df['kategori_usia'] == age_cat) & (combined_df['Type'] == 'Predicted')]
        if not pred_data.empty:
            fig.add_trace(go.Scatter(
                x=pred_data['id_tahun'],
                y=pred_data['total'],
                mode='lines+markers',
                name=f'{age_cat} (Predicted)',
                line=dict(color=age_colors[age_cat], width=3, dash='dash'),
                marker=dict(size=8, symbol='diamond')
            ))

    fig.update_layout(
        title='Trend Jumlah Penduduk per Kelompok Umur (Historikal & Prediksi)',
        xaxis_title='Tahun',
        yaxis_title='Jumlah Penduduk',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    # Display prediction table
    st.header("Tabel Prediksi Detail")
    
    def format_value(x, is_pct=False):
        if pd.isna(x):
            return "-"
        if is_pct:
            return f"{x:.1f}%"
        return f"{x:,.0f}"
    
    def style_negative_positive(val):
        if pd.isna(val):
            return ''
        if isinstance(val, str) and '%' in val:
            val = float(val.replace('%', ''))
        if val < 0:
            return 'color: #e74c3c'
        elif val > 0:
            return 'color: #2ecc71'
        return ''

    # Style the prediction table
    styled_pred_df = pred_df.copy()
    for col in ['Total', 'Laki-laki', 'Perempuan']:
        styled_pred_df[col] = styled_pred_df[col].apply(lambda x: format_value(x))
    for col in ['% Δ Total', '% Δ Laki', '% Δ Perempuan']:
        styled_pred_df[col] = styled_pred_df[col].apply(lambda x: format_value(x, is_pct=True))

    st.dataframe(
        styled_pred_df.style.applymap(style_negative_positive, subset=['% Δ Total', '% Δ Laki', '% Δ Perempuan']),
        use_container_width=True
    )

    st.write("*% Δ Laki-laki : presentase perubahan jumlah laki-laki dari data sebelumnya")
    st.write("*% Δ Perempuan : presentase perubahan jumlah perempuan dari data sebelumnya")
    st.write("*% Δ Total : presentase perubahan jumlah penduduk (laki-laki dan perempuan) dari data sebelumnya")