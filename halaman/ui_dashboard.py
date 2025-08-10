import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from supabase import create_client, Client
from model import fetch_data, train_svm_model, predict_population
import os

# Koneksi ke Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv('SUPABASE_KEY')  # Menggunakan environment variable untuk keamanan
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def app():
    # ======= DETAIL TABLE =======
    st.header("Data Historis")

    # Define all possible columns we might want to display
    possible_columns = {
        "id_tahun": "Tahun",
        "laki_laki": "Laki-laki",
        "perempuan": "Perempuan",
        "jumlah_penduduk": "Total Penduduk"
    }

    # Check which columns actually exist in our DataFrame
    available_cols = {col: name for col, name in possible_columns.items() if col in df.columns}

    if not available_cols:
        st.error("No valid population data columns found in the DataFrame!")
        return

    # Calculate percentage changes only for columns that exist
    if "laki_laki" in df.columns:
        df["% Perubahan Laki_laki"] = df["laki_laki"].pct_change() * 100
        available_cols["% Perubahan Laki_laki"] = "% Δ Laki-laki"

    if "perempuan" in df.columns:
        df["% Perubahan Perempuan"] = df["perempuan"].pct_change() * 100
        available_cols["% Perubahan Perempuan"] = "% Δ Perempuan"

    if "jumlah_penduduk" in df.columns:
        df["% Perubahan Jumlah Penduduk"] = df["jumlah_penduduk"].pct_change() * 100
        available_cols["% Perubahan Jumlah Penduduk"] = "% Δ Total"

    # Create the display DataFrame with only available columns
    final_df = df[list(available_cols.keys())].rename(columns=available_cols)

    # Convert all values to strings
    final_df_str = final_df.astype(str)

    # Apply formatting to string values
    for col in final_df_str.columns:
        if col in ["Laki-laki", "Perempuan", "Total Penduduk"]:
            final_df_str[col] = final_df_str[col].apply(lambda x: f"{float(x):,.0f}" if x.replace('.','',1).isdigit() else x)
        elif col in ["% Δ Laki-laki", "% Δ Perempuan", "% Δ Total"]:
            final_df_str[col] = final_df_str[col].apply(lambda x: f"{float(x):+.1f}%" if x.replace('.','',1).replace('-','',1).isdigit() else x)

    # Apply styling
    def color_negative_red(val):
        if isinstance(val, str) and val.startswith('-'):
            return 'color: #e74c3c'
        elif isinstance(val, str) and val[0].isdigit():
            return 'color: #ffffff'
        if isinstance(val, str) and val.startswith('+'):
            return 'color: #2ecc71'
        return ''

    styled_df = final_df_str.style.map(color_negative_red)

    # Display the table
    st.dataframe(
        styled_df,
        use_container_width=True,
        hide_index=True
    )

    st.write("*% Δ Laki-laki : presentase perubahan jumlah laki-laki dari data sebelumnya")
    st.write("*% Δ Perempuan : presentase perubahan jumlah perempuan dari data sebelumnya")
    st.write("*% Δ Total : presentase perubahan jumlah penduduk (laki-laki dan perempuan) dari data sebelumnya")