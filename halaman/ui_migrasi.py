import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from model import fetch_data, train_svm_model, predict_population

def app():
    # ======= TABEL DETAIL =======
    st.header("Detail Data Historis")
    
    # Format tabel
    display_df = df.rename(columns={
        "id_tahun": "Tahun",
        "migrasi_masuk": "Migrasi Masuk",
        "% Perubahan Masuk": "% Δ Masuk",
        "migrasi_keluar": "Migrasi Keluar", 
        "% Perubahan Keluar": "% Δ Keluar"
    })
    
    # Konversi ke string dengan format yang tepat
    display_df['Tahun'] = display_df['Tahun'].astype(str)  # Pastikan tahun sebagai string
    display_df['Migrasi Masuk'] = display_df['Migrasi Masuk'].apply(lambda x: f"{int(x):,}")
    display_df['Migrasi Keluar'] = display_df['Migrasi Keluar'].apply(lambda x: f"{int(x):,}")
    display_df['% Δ Masuk'] = display_df['% Δ Masuk'].apply(lambda x: f"{x:+.1f}%" if pd.notnull(x) else "")
    display_df['% Δ Keluar'] = display_df['% Δ Keluar'].apply(lambda x: f"{x:+.1f}%" if pd.notnull(x) else "")
    
    # Fungsi styling khusus
    def style_negative_positive(val):
        if not isinstance(val, str) or len(val) == 0:
            return ''
        if (val.startswith('+') or val[0].isdigit()) and '%' in val:
            color = '#2ecc71'  # Hijau untuk positif
        elif val.startswith('-') and '%' in val:
            color = '#e74c3c'  # Merah untuk negatif
        else:
            return ''
        return f'color: {color}'
    
    # Terapkan styling
    styled_df = (
        display_df.style
        .map(style_negative_positive, subset=["% Δ Masuk", "% Δ Keluar"])
        .format(None, na_rep="")  # Handle nilai kosong
    )
    
    st.dataframe(
        styled_df,
        use_container_width=True,
        hide_index=True
    )

    st.write("*% Δ Masuk : presentase perubahan jumlah penduduk masuk dari data sebelumnya")
    st.write("*% Δ Keluar : presentase perubahan jumlah penduduk keluar dari data sebelumnya")
    