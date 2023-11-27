import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Judul Aplikasi
st.title('Prediksi Tingkat Obesitas')

# Deskripsi
st.write('Aplikasi ini dapat memprediksi tingkat obesitas seseorang berdasarkan fitur-fitur tertentu.')

# Load model yang sudah disimpan
with open('best_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Fungsi untuk melakukan prediksi
def predict_obesity(age, height, weight, fcvc, ncp, ch2o, faf, tue, gender, family_history):
    # Encoding gender dan family history
    gender_encoded = 1 if gender == 'Male' else 0
    family_history_encoded = 1 if family_history == 'Yes' else 0

    input_data = np.array([age, height, weight, fcvc, ncp, ch2o, faf, tue, gender_encoded, family_history_encoded]).reshape(1, -1)
    prediction = model.predict(input_data)
    return prediction[0]

# Input fitur
st.sidebar.title('Masukkan Data Pasien')
age = st.sidebar.number_input('Usia', min_value=0)
height = st.sidebar.number_input('Tinggi (cm)', min_value=0)
weight = st.sidebar.number_input('Berat (kg)', min_value=0)
fcvc = st.sidebar.number_input('Frekuensi Makan Buah', min_value=0.0, max_value=1.0, step=0.01)
ncp = st.sidebar.number_input('Frekuensi Makan Sayuran', min_value=0.0, max_value=1.0, step=0.01)
ch2o = st.sidebar.number_input('Konsumsi Air (L)', min_value=0.0)
faf = st.sidebar.number_input('Aktivitas Fisik (jam per hari)', min_value=0.0)
tue = st.sidebar.number_input('Waktu Layar (jam per hari)', min_value=0.0)
gender = st.sidebar.selectbox('Jenis Kelamin', ['Male', 'Female'])
family_history = st.sidebar.selectbox('Riwayat Keluarga dengan Kegemukan', ['Yes', 'No'])

# Tombol untuk prediksi
if st.sidebar.button('Prediksi'):
    prediction = predict_obesity(age, height, weight, fcvc, ncp, ch2o, faf, tue, gender, family_history)
    st.sidebar.write(f'Hasil Prediksi: {prediction}')

# # Informasi tambahan
# st.sidebar.write("Tingkat obesitas diprediksi sebagai berikut:")
# st.sidebar.write("0: Underweight")
# st.sidebar.write("1: Normal Weight")
# st.sidebar.write("2: Overweight Level I")
# st.sidebar.write("3: Overweight Level II")
# st.sidebar.write("4: Obesity Type I")
# st.sidebar.write("5: Obesity Type II")
# st.sidebar.write("6: Obesity Type III")
