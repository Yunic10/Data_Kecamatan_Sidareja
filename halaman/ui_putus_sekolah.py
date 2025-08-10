import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from model import fetch_data, train_svm_model, predict_population

def app():
    # ======= DATA PREPARATION ======= 
    df = fetch_data(
        table_name="putus_sekolah",
        feature_columns=["id_tahun"],
        target_columns=["jumlah_putus_sekolah"]
    ).sort_values("id_tahun")
    
    # Hitung perubahan
    df["% Perubahan"] = df["jumlah_putus_sekolah"].pct_change() * 100
    
        # ======= TABEL DETAIL =======
    st.header("Detail Data Historis")

    # Format tabel
    display_df = df.rename(columns={
        "id_tahun": "Tahun",
        "jumlah_putus_sekolah": "Jumlah Anak Putus Sekolah",
        "% Perubahan": "Perubahan (%)"
    })

    # Konversi ke string dengan format yang tepat
    display_df['Tahun'] = display_df['Tahun'].astype(str)  # Pastikan tahun sebagai string
    display_df['Jumlah Anak Putus Sekolah'] = display_df['Jumlah Anak Putus Sekolah'].apply(lambda x: f"{int(x):,}")
    display_df['Perubahan (%)'] = display_df['Perubahan (%)'].apply(lambda x: f"{x:+.1f}%" if pd.notnull(x) else "")
    
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
        .map(style_negative_positive, subset=["Perubahan (%)"])
        .format(None, na_rep="")  # Handle nilai kosong
        .set_properties(**{
            'text-align': 'center',
            'padding': '8px 12px',
            'font-family': 'Arial, sans-serif'
        })
        .set_table_styles([{
            'selector': 'thead th',
            'props': [
                ('background-color', '#2c3e50'),
                ('color', 'white'),
                ('font-weight', 'bold'),
                ('text-align', 'center')
            ]
        }])
    )
    
    st.dataframe(
        styled_df,
        use_container_width=True,
        hide_index=True
    )

    st.write("*% perubahan : presentase perubahan jumlah anak putus sekolah dari data sebelumnya")