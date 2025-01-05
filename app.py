import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Load model
model = joblib.load("model.pkl")

# Set title
st.title("Prediksi Data dengan Streamlit")
st.write("""Pengingat: Untuk pengujian satu data anda hanya perlu 
         diminta untuk memasukkan feature-feature tertentu yang 
         feature tersebut dipilih setelah dilakukan analisis""")

# Fungsi preprocessing
def preprocess(data):
    data = data[['Tuition_fees_up_to_date', 'Curricular_units_2nd_sem_grade', 'Daytime_evening_attendance', 'Scholarship_holder']]

    feature_numerik_kontinus = ['Curricular_units_2nd_sem_grade']
    feature_numerik_diskrit = ['Tuition_fees_up_to_date', 'Daytime_evening_attendance', 'Scholarship_holder']

    # Normalisasi
    norm_minmax = MinMaxScaler()
    norm_zscore = StandardScaler()

    data[feature_numerik_diskrit] = norm_minmax.fit_transform(data[feature_numerik_diskrit])
    data[feature_numerik_kontinus] = norm_zscore.fit_transform(data[feature_numerik_kontinus])

    return data

# Pilihan input data
st.sidebar.title("Pilih Metode Input Data")
input_method = st.sidebar.radio("Metode Input:", ("Manual", "Upload CSV"))

if input_method == "Manual":
    # Input manual
    Tuition_fees_up_to_date = st.sidebar.selectbox("Tuition Fees Up To Date: (Yes = 1 | No = 0)", [0, 1])
    Curricular_units_2nd_sem_grade = st.sidebar.number_input("Curricular Units 2nd Sem Grade:", min_value=0, max_value=24, step=1)
    Daytime_evening_attendance = st.sidebar.selectbox("Daytime/Evening Attendance: (Evening = 1 | Daytime = 0 | )", [1, 0])
    Scholarship_holder = st.sidebar.selectbox("Scholarship Holder: (Yes = 1 | No = 0)", [1, 0])

    # Tombol prediksi
    if st.sidebar.button("Prediksi"):
        # Buat dataframe input
        input_data = pd.DataFrame({
            'Tuition_fees_up_to_date': [Tuition_fees_up_to_date],
            'Curricular_units_2nd_sem_grade': [Curricular_units_2nd_sem_grade],
            'Daytime_evening_attendance': [Daytime_evening_attendance],
            'Scholarship_holder': [Scholarship_holder]
        })

        # Preprocessing data
        preprocessed_data = preprocess(input_data)

        # Prediksi
        prediction = model.predict(preprocessed_data)

        # Tampilkan hasil
        st.write("### Hasil Prediksi")
        st.write(f"Prediksi: {'Graduate' if prediction[0] == 1 else 'Dropout'}")

elif input_method == "Upload CSV":
    # Input melalui upload file CSV
    uploaded_file = st.sidebar.file_uploader("Upload file CSV:", type="csv")

    if uploaded_file is not None:
        # Baca file CSV
        data = pd.read_csv(uploaded_file,  sep = ";")

        # Preprocessing data
        preprocessed_data = preprocess(data)

        # Prediksi
        prediction = model.predict(preprocessed_data)

        # Tampilkan hasil
        st.write("### Hasil Prediksi")
        data['Prediction'] = ["Graduate" if i == 1 else "Dropout" for i in prediction]
        st.write(data)
