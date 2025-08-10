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
        "status_kawin": "Kawin (Jiwa)",
        "% Perubahan Kawin": "% Δ Kawin",
        "cerai_hidup": "Cerai Hidup (Jiwa)", 
        "% Perubahan Cerai": "% Δ Cerai"
    })
    
    # Konversi ke string dengan format yang tepat
    display_df['Tahun'] = display_df['Tahun'].astype(str)  # Pastikan tahun sebagai string
    display_df['Kawin (Jiwa)'] = display_df['Kawin (Jiwa)'].apply(lambda x: f"{int(x):,}")
    display_df['Cerai Hidup (Jiwa)'] = display_df['Cerai Hidup (Jiwa)'].apply(lambda x: f"{int(x):,}")
    display_df['% Δ Kawin'] = display_df['% Δ Kawin'].apply(lambda x: f"{x:+.1f}%" if pd.notnull(x) else "")
    display_df['% Δ Cerai'] = display_df['% Δ Cerai'].apply(lambda x: f"{x:+.1f}%" if pd.notnull(x) else "")
    
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
        .map(style_negative_positive, subset=["% Δ Kawin", "% Δ Cerai"])
        .format(None, na_rep="")  # Handle nilai kosong
    )
    
    st.dataframe(
        styled_df,
        use_container_width=True,
        hide_index=True
    )

    st.write("*% Δ Kawin : presentase perubahan jumlah status kawin dari data sebelumnya")
    st.write("*% Δ Cerai : presentase perubahan jumlah status cerai dari data sebelumnya")
    