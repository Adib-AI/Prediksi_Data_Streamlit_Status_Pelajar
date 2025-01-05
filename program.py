import pandas as pd
import numpy as np
import joblib
import argparse
from sklearn.preprocessing import StandardScaler, MinMaxScaler


model = joblib.load("model.pkl")

def preproces(data):
    data = data[['Tuition_fees_up_to_date', 'Curricular_units_2nd_sem_grade','Daytime_evening_attendance', 'Scholarship_holder']]
    
    feature_numerik_kontinus = data.select_dtypes(include='float').columns
    feature_numerik_diskrit = data.select_dtypes(include = 'integer').columns

    #Normalisasi
    norm_minmax = MinMaxScaler()
    norm_zscore = StandardScaler()

    data[feature_numerik_diskrit] = norm_minmax.fit_transform(data[feature_numerik_diskrit])
    data[feature_numerik_kontinus] = norm_zscore.fit_transform(data[feature_numerik_kontinus])

    return data

def main(input_data):
    if isinstance(input_data, str):
        data = pd.read_csv(input_data, sep = ';') 
    else:
        data = pd.DataFrame(input_data)

    data_preprocess = preproces(data)
    print("Preprocessing berhasil..")

    predict = model.predict(data_preprocess)

    data["Prediksi"] = predict

    data.to_csv("Hasil_prediksi.csv", index = False)
    print("Prediksi berhasil.. dan file telah dibuat")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Proyek Akhir: enyelesaikan Permasalahan Institusi Pendidikan")
    parser.add_argument('--input', type = str, required=True, help="Path to the input CSV file.")
    args = parser.parse_args()

    #Pengujian
    main(args.input)