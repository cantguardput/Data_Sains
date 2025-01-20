import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import io
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

st.set_page_config(
    page_title="Aplikasi Prediksi Jantung",
    page_icon="🩺",
    layout="wide"
)

# Load dataset
@st.cache_data
def load_data():
    file_path = "heart (1).csv"
    data = pd.read_csv(file_path)
    data = data.dropna()
    features = data.columns[:-1].tolist()
    target = data.columns[-1]
    return data, features, target

def show_dataset_info(data):
    st.subheader("Informasi Dataset")
    
    # Basic dataset information
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"Total Data: {len(data)}")
    with col2:
        st.info(f"Data Training: {int(len(data) * 0.8)}")
    with col3:
        st.info(f"Data Testing: {int(len(data) * 0.2)}")
    
    # Dataset distribution
    st.subheader("Distribusi Target")
    target_dist = data['target'].value_counts()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(['Tidak Rawan', 'Rawan'], 
           [target_dist[0], target_dist[1]], 
           color=['green', 'red'])
    plt.title("Distribusi Kasus Penyakit Jantung")
    plt.ylabel("Jumlah Kasus")
    st.pyplot(fig)

def show_dashboard(data, reg_metrics, elm_metrics):
    st.title("Dashboard Metode Prediksi Penyakit Jantung")
    
    # Introduction
    st.write("""
    Aplikasi ini menggunakan dua metode machine learning untuk memprediksi risiko penyakit jantung:
    Multiple Linear Regression (MLR) dan Extreme Learning Machine (ELM).
    """)
    
    # Method Explanations
    st.header("Penjelasan Metode")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Multiple Linear Regression (MLR)")
        st.write("""
        Multiple Linear Regression adalah metode statistik yang memodelkan hubungan linear 
        antara beberapa variabel independen dengan satu variabel dependen. Dalam konteks ini:
        
        - Menganalisis hubungan linear antara faktor risiko dengan kemungkinan penyakit jantung
        - Cocok untuk memahami pengaruh setiap faktor risiko
        - Mudah diinterpretasi dan dijelaskan
        """)
        
    with col2:
        st.subheader("Extreme Learning Machine (ELM)")
        st.write("""
        Extreme Learning Machine adalah jenis neural network dengan single hidden layer 
        yang memiliki kecepatan pembelajaran yang sangat cepat. Karakteristik:
        
        - Mampu menangkap pola non-linear dalam data
        - Proses pembelajaran yang lebih cepat
        - Performa yang baik untuk klasifikasi
        """)
    
    # Performance Metrics Visualization
    st.header("Performa Model")
    
    # Prepare metrics for visualization
    metrics_comparison = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
        'MLR': [reg_metrics['Accuracy'], reg_metrics['Precision'], 
                reg_metrics['Recall'], reg_metrics['F1 Score']],
        'ELM': [elm_metrics['Accuracy'], elm_metrics['Precision'], 
                elm_metrics['Recall'], elm_metrics['F1 Score']]
    })
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(metrics_comparison['Metric']))
    width = 0.35
    
    ax.bar(x - width/2, metrics_comparison['MLR'], width, label='MLR', color='skyblue')
    ax.bar(x + width/2, metrics_comparison['ELM'], width, label='ELM', color='lightgreen')
    
    ax.set_ylabel('Score')
    ax.set_title('Perbandingan Metrik Performa Model')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_comparison['Metric'])
    ax.legend()
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Detailed Metrics Table
    st.subheader("Detail Metrik Performa")
    metrics_df = pd.DataFrame({
        'MLR': reg_metrics,
        'ELM': elm_metrics
    }).round(3)
    st.dataframe(metrics_df)
    
    # Recommendation
    st.header("Rekomendasi Penggunaan")
    st.write("""
    Berdasarkan performa kedua model:
    
    1. **Multiple Linear Regression (MLR)** cocok digunakan ketika:
       - Membutuhkan interpretasi yang mudah dipahami
       - Ingin melihat pengaruh langsung setiap faktor risiko
       - Memerlukan prediksi yang stabil dan konsisten
    
    2. **Extreme Learning Machine (ELM)** cocok digunakan ketika:
       - Membutuhkan akurasi prediksi yang lebih tinggi
       - Data memiliki pola non-linear yang kompleks
       - Kecepatan pemrosesan menjadi prioritas
    """)

def main():
    # Load data and prepare models
    data, features, target = load_data()
    X = data[features]
    y = data[target]
    
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Prepare models
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    reg_model = LinearRegression().fit(X_train, y_train)
    elm_model = MLPRegressor(hidden_layer_sizes=(50,), max_iter=500, random_state=42).fit(X_train_scaled, y_train)
    
    # Calculate metrics
    reg_pred = reg_model.predict(X_test)
    elm_pred = elm_model.predict(X_test_scaled)
    
    reg_metrics = {
        'MSE': mean_squared_error(y_test, reg_pred),
        'R² Score': reg_model.score(X_test, y_test),
        'Accuracy': accuracy_score(y_test, reg_pred.round()),
        'Precision': precision_score(y_test, reg_pred.round()),
        'Recall': recall_score(y_test, reg_pred.round()),
        'F1 Score': f1_score(y_test, reg_pred.round())
    }
    
    elm_metrics = {
        'MSE': mean_squared_error(y_test, elm_pred),
        'R² Score': elm_model.score(X_test_scaled, y_test),
        'Accuracy': accuracy_score(y_test, elm_pred.round()),
        'Precision': precision_score(y_test, elm_pred.round()),
        'Recall': recall_score(y_test, elm_pred.round()),
        'F1 Score': f1_score(y_test, elm_pred.round())
    }
    
    # Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Pilih Halaman:", ["Dashboard", "Dataset Info", "Prediksi"])
    
    if page == "Dashboard":
        show_dashboard(data, reg_metrics, elm_metrics)
    elif page == "Dataset Info":
        show_dataset_info(data)
    else:  # Prediction page
        st.title("Prediksi Penyakit Jantung")
        
        # Method Selection
        selected_method = st.sidebar.selectbox(
            "Pilih Metode Prediksi",
            ["Pilih Metode", "Multiple Regression", "ELM"]
        )
        
        if selected_method == "Pilih Metode":
            st.info("Silakan pilih metode prediksi di sidebar untuk memulai.")
            return
            
        # Patient Information
        st.sidebar.title("Data Pasien")
        patient_name = st.sidebar.text_input("Nama Pasien", "")
        
        # Medical Parameters
        st.sidebar.title("Data Medical")
        user_input_dict = {}
        user_input = []
        
        for feature in features:
            if feature == "sex":
                gender = st.sidebar.selectbox("sex", ["Laki-laki", "Perempuan"])
                value = 1 if gender == "Laki-laki" else 0
                user_input_dict[feature] = "Laki-laki" if value == 1 else "Perempuan"
            elif feature == "age":
                min_val, max_val = int(X[feature].min()), int(X[feature].max())
                value = st.sidebar.number_input(f"{feature} ({min_val} - {max_val})", 
                                              min_value=min_val, max_value=max_val, 
                                              value=int((min_val + max_val) / 2), step=1)
                user_input_dict[feature] = value
            else:
                min_val, max_val = float(X[feature].min()), float(X[feature].max())
                value = st.sidebar.number_input(f"{feature} ({min_val} - {max_val})", 
                                              min_value=min_val, max_value=max_val, 
                                              value=(min_val + max_val) / 2)
                user_input_dict[feature] = value
            user_input.append(value)
        
        user_input = np.array(user_input).reshape(1, -1)
        
        if st.button("Mulai Prediksi"):
            if not patient_name:
                st.error("Harap Diisi Nama Pasien Sebelum Memulai Prediksi.")
                return
                
            st.header(f"Hasil Analisis untuk {patient_name}")
            
            # Get predictions
            reg_prediction = reg_model.predict(user_input)[0]
            user_input_scaled = scaler.transform(user_input)
            elm_prediction = elm_model.predict(user_input_scaled)[0]
            
            predictions = {
                "Multiple Regression": reg_prediction,
                "ELM": elm_prediction
            }
            
            # Display selected method first
            st.subheader(f"Hasil Prediksi dengan {selected_method}")
            selected_prediction = reg_prediction if selected_method == "Multiple Regression" else elm_prediction
            
            if selected_prediction >= 0.5:
                st.error("🩺 Rawan Terkena Penyakit Jantung")
            else:
                st.success("✅ Tidak Rawan Terkena Penyakit Jantung")
            st.write(f"Nilai Prediksi: {selected_prediction:.3f}")
            
            # Comparison with other method
            st.subheader("Perbandingan dengan Metode Lain")
            other_method = "ELM" if selected_method == "Multiple Regression" else "Multiple Regression"
            other_prediction = elm_prediction if selected_method == "Multiple Regression" else reg_prediction
            
            if other_prediction >= 0.5:
                st.error(f"🩺 {other_method}: Rawan Terkena Penyakit Jantung")
            else:
                st.success(f"✅ {other_method}: Tidak Rawan Terkena Penyakit Jantung")
            st.write(f"Nilai Prediksi {other_method}: {other_prediction:.3f}")
            
            # Generate PDF report
            pdf_buffer = create_pdf_report(predictions, 
                                         {"Multiple Regression": reg_metrics, "ELM": elm_metrics},
                                         user_input_dict, 
                                         patient_name,
                                         selected_method)
            
            # Download button for PDF
            st.download_button(
                label="Download Laporan PDF",
                data=pdf_buffer,
                file_name=f"prediksi_penyakit_jantung_{patient_name.replace(' ', '_')}.pdf",
                mime="application/pdf"
            )

if __name__ == "__main__":
    main()
