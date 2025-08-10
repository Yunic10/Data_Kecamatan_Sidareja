import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from model import fetch_data, train_svm_model, predict_population

def style_negative_positive(val):
    if not isinstance(val, str) or len(val) == 0:
        return ''
    if (val.startswith('+') or val[0].isdigit()) and '%' in val:
        color = '#2ecc71'
    elif val.startswith('-') and '%' in val:
        color = '#e74c3c'
    else:
        return ''
    return f'color: {color}'

def app(): 
    # ======= HISTORICAL DATA TABLE =======
    st.header("Data Historis")
    
    # Format the historical data
    hist_df = df.copy()
    hist_df = hist_df.rename(columns={
        'id_tahun': 'Tahun',
        'pria': 'Pria',
        'wanita': 'Wanita',
        'jumlah_kepala_keluarga': 'Total Kepala Keluarga',
        '% Perubahan Pria': '% Δ Pria',
        '% Perubahan Wanita': '% Δ Wanita',
        '% Perubahan jumlah_kepala_keluarga': '% Δ Total'
    })
    
    # Convert to formatted strings
    hist_df['Tahun'] = hist_df['Tahun'].astype(str)  
    hist_df['Pria'] = hist_df['Pria'].apply(lambda x: f"{x:,.0f}")
    hist_df['Wanita'] = hist_df['Wanita'].apply(lambda x: f"{x:,.0f}")
    hist_df['Total Kepala Keluarga'] = hist_df['Total Kepala Keluarga'].apply(lambda x: f"{x:,.0f}")
    
    # Apply percentage formatting
    pct_cols = ['% Δ Pria', '% Δ Wanita', '% Δ Total']
    for col in pct_cols:
        if col in hist_df.columns:
            hist_df[col] = hist_df[col].apply(lambda x: f"{x:+.1f}%" if pd.notnull(x) else "")
    
    # Display the table with styling
    st.dataframe(
        hist_df.style.map(style_negative_positive, subset=pct_cols),
        use_container_width=True,
        hide_index=True
    )

    st.write("*% Δ Pria : presentase perubahan jumlah pria dari data sebelumnya")
    st.write("*% Δ Wanita : presentase perubahan jumlah wanita dari data sebelumnya")
    st.write("*% Δ Total : presentase perubahan jumlah kepala keluarga (pria dan wanita) dari data sebelumnya")