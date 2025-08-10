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
    st.title("Prediksi Populasi Kecamatan Sidareja")

    # Ambil data tahunan untuk grafik
    df = fetch_data(
        table_name="penduduk_tahunan", 
        feature_columns=["id_tahun"], 
        target_columns=["jumlah_penduduk", "laki_laki", "perempuan"]
    ).sort_values("id_tahun")
    
    # Calculate jumlah_penduduks and changes
    df['Jumlah Penduduk'] = df['laki_laki'] + df['perempuan']
    df["% Perubahan Laki_laki"] = df["laki_laki"].pct_change() * 100
    df["% Perubahan Perempuan"] = df["perempuan"].pct_change() * 100
    df["% Perubahan Jumlah Penduduk"] = df["jumlah_penduduk"].pct_change() * 100
    
    # Train model for each category
    models = {}
    metrics = {}
    for target in ['laki_laki', 'perempuan', 'jumlah_penduduk']:
        model, mae, mape, r2 = train_svm_model(
            table_name="penduduk_tahunan",
            feature_columns=["id_tahun"],
            target_column=target
        )
        models[target] = model
        metrics[target] = {'MAPE': mape, 'R²': r2}

    # ======= PREDICTION LOGIC =======
    last_year = df['id_tahun'].max()
    next_years = np.array([last_year + 1, last_year + 2, last_year + 3]).reshape(-1, 1)  # 3 tahun prediksi

    # Mendapatkan prediksi untuk 3 tahun
    predictions = {
        'laki_laki': models['laki_laki'].predict(next_years),
        'perempuan': models['perempuan'].predict(next_years)
    }

    # Menghitung perubahan persentase
    last_values = df[df['id_tahun'] == last_year].iloc[0]
    changes = {
        'laki_laki': (predictions['laki_laki'] - last_values['laki_laki']) / last_values['laki_laki'] * 100,
        'perempuan': (predictions['perempuan'] - last_values['perempuan']) / last_values['perempuan'] * 100,
        'jumlah_penduduk': ((predictions['laki_laki'] + predictions['perempuan']) - last_values['jumlah_penduduk']) / last_values['jumlah_penduduk'] * 100
    }

    # ======= PREDICTION DISPLAY =======
    st.header("Prediksi 3 Tahun ke Depan")
    cols = st.columns(3)
    with cols[0]:
        st.metric(
            f"Prediksi Laki-laki {last_year+1}", 
            f"{predictions['laki_laki'][0]:,.0f}",
            delta=f"{changes['laki_laki'][0]:+.1f}%"
        )
        st.metric(
            f"Prediksi Laki-laki {last_year+2}", 
            f"{predictions['laki_laki'][1]:,.0f}",
            delta=f"{changes['laki_laki'][1]:+.1f}%"
        )
        st.metric(
            f"Prediksi Laki-laki {last_year+3}", 
            f"{predictions['laki_laki'][2]:,.0f}",
            delta=f"{changes['laki_laki'][2]:+.1f}%"
        )

    with cols[1]:
        st.metric(
            f"Prediksi Perempuan {last_year+1}",
            f"{predictions['perempuan'][0]:,.0f}",
            delta=f"{changes['perempuan'][0]:+.1f}%"
        )
        st.metric(
            f"Prediksi Perempuan {last_year+2}",
            f"{predictions['perempuan'][1]:,.0f}",
            delta=f"{changes['perempuan'][1]:+.1f}%"
        )
        st.metric(
            f"Prediksi Perempuan {last_year+3}",
            f"{predictions['perempuan'][2]:,.0f}",
            delta=f"{changes['perempuan'][2]:+.1f}%"
        )

    with cols[2]:
        st.metric(
            f"Total Penduduk {last_year+1}",
            f"{predictions['laki_laki'][0] + predictions['perempuan'][0]:,.0f}",
            delta=f"{changes['jumlah_penduduk'][0]:+.1f}%"
        )
        st.metric(
            f"Total Penduduk {last_year+2}",
            f"{predictions['laki_laki'][1] + predictions['perempuan'][1]:,.0f}",
            delta=f"{changes['jumlah_penduduk'][1]:+.1f}%"
        )
        st.metric(
            f"Total Penduduk {last_year+3}",
            f"{predictions['laki_laki'][2] + predictions['perempuan'][2]:,.0f}",
            delta=f"{changes['jumlah_penduduk'][2]:+.1f}%"
        )

    # ======= VISUALIZATION =======
    viz_df = df.tail(8).copy()
    viz_df['Type'] = 'Historical'

    # Membuat pred_df dengan panjang yang konsisten (3 tahun)
    pred_df = pd.DataFrame({
        'id_tahun': next_years.flatten(),
        'laki_laki': predictions['laki_laki'],
        'perempuan': predictions['perempuan'],
        'jumlah_penduduk': predictions['laki_laki'] + predictions['perempuan'],
        'Type': ['Predicted'] * 3
    })

    viz_df = pd.concat([viz_df, pred_df])    

    # Create line chart for trends
    fig_line = px.line(
        viz_df,
        x="id_tahun",
        y="jumlah_penduduk",
        color="Type",
        markers=True,
        title=f"Trend Line Jumlah Penduduk",
        labels={"id_tahun": "Tahun", "jumlah_penduduk": "Jumlah Penduduk"},
        color_discrete_map={"Historical": "#2ecc71", "Predicted": "#e74c3c"}
    )
    fig_line.update_layout(
        showlegend=True,
        xaxis_title="Tahun",
        yaxis_title="Jumlah Penduduk"
    )

    # Display charts
    st.plotly_chart(fig_line, use_container_width=True)
    
    # ===== NEW VISUALIZATIONS: POPULATION GROWTH ANALYSIS =====
    st.header("Analisis Pertumbuhan Penduduk")
    
    # 1. Growth Rate Over Time
    growth_df = df[df['id_tahun'] > df['id_tahun'].min()]  # Exclude first year with no growth data
    
    fig_growth = px.line(
        growth_df,
        x='id_tahun',
        y=['% Perubahan Laki_laki', '% Perubahan Perempuan', '% Perubahan Jumlah Penduduk'],
        labels={'value': 'Persentase Pertumbuhan (%)', 'id_tahun': 'Tahun', 'variable': 'Kategori'},
        title='Persentase Pertumbuhan Penduduk per Tahun',
        markers=True
    )
    fig_growth.update_layout(
        yaxis_title="Pertumbuhan (%)",
        legend_title="Kategori",
        hovermode="x unified"
    )
    st.plotly_chart(fig_growth, use_container_width=True)
    
    # 2. Average Growth Rate Metrics
    st.subheader("Rata-rata Pertumbuhan Tahunan")
    
    # Calculate averages
    avg_growth_total = df['% Perubahan Jumlah Penduduk'].mean()
    avg_growth_male = df['% Perubahan Laki_laki'].mean()
    avg_growth_female = df['% Perubahan Perempuan'].mean()
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Penduduk", f"{avg_growth_total:.2f}%")
    col2.metric("Laki-laki", f"{avg_growth_male:.2f}%")
    col3.metric("Perempuan", f"{avg_growth_female:.2f}%")
    
    # 4. Population Density and Projection
    st.subheader("Proyeksi Kepadatan Penduduk")
    
    # Assuming we have area data (in km²)
    AREA_SIDAREJA = 49.31  # Example area in km²
    
    # Calculate density
    df['Kepadatan Penduduk'] = df['jumlah_penduduk'] / AREA_SIDAREJA
    current_density = df.iloc[-1]['Kepadatan Penduduk']
    
    # Projected density
    projected_population = predictions['laki_laki'][2] + predictions['perempuan'][2]
    projected_density = projected_population / AREA_SIDAREJA
    
    col1, col2 = st.columns(2)
    col1.metric("Kepadatan Penduduk Saat Ini", 
               f"{current_density:,.0f} jiwa/km²",
               help=f"Luas wilayah: {AREA_SIDAREJA} km²")
    
    col2.metric("Proyeksi Kepadatan 3 Tahun Mendatang", 
               f"{projected_density:,.0f} jiwa/km²",
               delta=f"{(projected_density - current_density):+.0f} jiwa/km²")

    # ======= MODEL PERFORMANCE =======
    st.header("Model Prediksi Berdasarkan MAPE dan R²")

    perf_df = pd.DataFrame(metrics).T.reset_index()
    perf_df.columns = ['Kategori', 'MAPE (%)', 'R²']

    # Convert performance metrics to strings with formatting
    perf_df_str = perf_df.copy()
    perf_df_str['MAPE (%)'] = perf_df_str['MAPE (%)'].apply(lambda x: f"{x:,.1f}%")
    perf_df_str['R²'] = perf_df_str['R²'].apply(lambda x: f"{x:.3f}")

    st.dataframe(
        perf_df_str,
        use_container_width=True,
        hide_index=True
    )
    
    st.write("*MAPE (Mean Absolute Precentage Error): Rata-rata persentase kesalahan prediksi dibandingkan nilai aktual. " \
    "Semakin kecil MAPE semakin akurat model")
    st.write("*R-squared: Mengukur seberapa baik model menjelaskan variasi data. " \
    "Semakin mendekati angka 1 semakin baik model menjelaskan variasi data")
    
    st.markdown("---")
    st.caption("© 2025 - Yudith Nico Priambodo")