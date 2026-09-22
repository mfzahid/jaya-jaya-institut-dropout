import pandas as pd
import numpy as np
import joblib
import os

# ============================================================
# Konstanta
# ============================================================
DATA_URL = (
    'https://raw.githubusercontent.com/dicodingacademy/dicoding_dataset'
    '/main/students_performance/data.csv'
)

# Mapping status ke label biner (hanya Dropout vs Graduate)
STATUS_BINARY = {'Graduate': 0, 'Dropout': 1}

# Fitur yang digunakan (semua kolom kecuali Status)
FEATURE_COLUMNS = [
    'Marital_status', 'Application_mode', 'Application_order', 'Course',
    'Daytime_evening_attendance', 'Previous_qualification',
    'Previous_qualification_grade', 'Nacionality', 'Mothers_qualification',
    'Fathers_qualification', 'Mothers_occupation', 'Fathers_occupation',
    'Admission_grade', 'Displaced', 'Educational_special_needs', 'Debtor',
    'Tuition_fees_up_to_date', 'Gender', 'Scholarship_holder',
    'Age_at_enrollment', 'International',
    'Curricular_units_1st_sem_credited', 'Curricular_units_1st_sem_enrolled',
    'Curricular_units_1st_sem_evaluations', 'Curricular_units_1st_sem_approved',
    'Curricular_units_1st_sem_grade', 'Curricular_units_1st_sem_without_evaluations',
    'Curricular_units_2nd_sem_credited', 'Curricular_units_2nd_sem_enrolled',
    'Curricular_units_2nd_sem_evaluations', 'Curricular_units_2nd_sem_approved',
    'Curricular_units_2nd_sem_grade', 'Curricular_units_2nd_sem_without_evaluations',
    'Unemployment_rate', 'Inflation_rate', 'GDP',
]


def load_model(model_path: str = 'model/best_model.pkl'):
    """Memuat model yang telah disimpan."""
    fallback_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        os.path.basename(model_path)
    )
    if not os.path.exists(model_path) and os.path.exists(fallback_path):
        model_path = fallback_path
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model tidak ditemukan di '{model_path}'.\n"
            "Jalankan notebook.ipynb terlebih dahulu untuk melatih dan menyimpan model."
        )
    return joblib.load(model_path)


def preprocess_input(data: dict) -> pd.DataFrame:
    """
    Preprocessing data input untuk prediksi.

    Parameters
    ----------
    data : dict
        Dictionary berisi fitur siswa (semua nilai numerik)

    Returns
    -------
    pd.DataFrame
        DataFrame yang sudah diproses dan siap untuk prediksi
    """
    df = pd.DataFrame([data])

    # Pastikan hanya kolom fitur yang digunakan, sesuai urutan training
    cols_available = [c for c in FEATURE_COLUMNS if c in df.columns]
    df = df[cols_available]

    # Isi nilai yang hilang dengan 0
    df = df.fillna(0)

    return df


def predict(data: dict, model_path: str = 'model/best_model.pkl') -> dict:
    """
    Memprediksi potensi dropout seorang siswa.

    Parameters
    ----------
    data : dict
        Dictionary berisi fitur siswa (lihat FEATURE_COLUMNS)
    model_path : str
        Path ke file model yang telah disimpan

    Returns
    -------
    dict
        Hasil prediksi:
        - prediction (int): 0 = Graduate, 1 = Dropout
        - label (str): Label prediksi
        - probability_graduate (float): Probabilitas lulus
        - probability_dropout (float): Probabilitas dropout
        - risk_level (str): Tingkat risiko (Low / Medium / High)
    """
    model = load_model(model_path)
    df = preprocess_input(data)

    prediction = int(model.predict(df)[0])
    probabilities = model.predict_proba(df)[0]
    prob_dropout = float(probabilities[1])

    if prob_dropout < 0.3:
        risk_level = "Low"
    elif prob_dropout < 0.6:
        risk_level = "Medium"
    else:
        risk_level = "High"

    return {
        'prediction': prediction,
        'label': 'Dropout' if prediction == 1 else 'Graduate',
        'probability_graduate': float(probabilities[0]),
        'probability_dropout': prob_dropout,
        'risk_level': risk_level
    }


def predict_batch(data_list: list, model_path: str = 'model/best_model.pkl') -> pd.DataFrame:
    """
    Prediksi batch untuk beberapa siswa sekaligus.

    Parameters
    ----------
    data_list : list of dict
        List berisi data siswa
    model_path : str
        Path ke file model

    Returns
    -------
    pd.DataFrame
        DataFrame dengan hasil prediksi untuk setiap siswa
    """
    results = []
    for i, student_data in enumerate(data_list):
        try:
            result = predict(student_data, model_path)
            result['student_index'] = i
            results.append(result)
        except Exception as e:
            results.append({
                'student_index': i,
                'prediction': None,
                'label': 'Error',
                'probability_graduate': None,
                'probability_dropout': None,
                'risk_level': 'Unknown',
                'error': str(e)
            })
    return pd.DataFrame(results)


# ============================================================
# Contoh penggunaan
# ============================================================
if __name__ == '__main__':

    # --- Contoh 1: Siswa Berisiko Tinggi Dropout ---
    siswa_risiko_tinggi = {
        'Marital_status': 1,
        'Application_mode': 17,
        'Application_order': 5,
        'Course': 9,
        'Daytime_evening_attendance': 1,
        'Previous_qualification': 1,
        'Previous_qualification_grade': 100.0,
        'Nacionality': 1,
        'Mothers_qualification': 1,
        'Fathers_qualification': 1,
        'Mothers_occupation': 5,
        'Fathers_occupation': 5,
        'Admission_grade': 100.0,
        'Displaced': 0,
        'Educational_special_needs': 0,
        'Debtor': 1,
        'Tuition_fees_up_to_date': 0,
        'Gender': 1,
        'Scholarship_holder': 0,
        'Age_at_enrollment': 35,
        'International': 0,
        'Curricular_units_1st_sem_credited': 0,
        'Curricular_units_1st_sem_enrolled': 6,
        'Curricular_units_1st_sem_evaluations': 6,
        'Curricular_units_1st_sem_approved': 0,
        'Curricular_units_1st_sem_grade': 0.0,
        'Curricular_units_1st_sem_without_evaluations': 0,
        'Curricular_units_2nd_sem_credited': 0,
        'Curricular_units_2nd_sem_enrolled': 6,
        'Curricular_units_2nd_sem_evaluations': 4,
        'Curricular_units_2nd_sem_approved': 0,
        'Curricular_units_2nd_sem_grade': 0.0,
        'Curricular_units_2nd_sem_without_evaluations': 2,
        'Unemployment_rate': 10.8,
        'Inflation_rate': 1.4,
        'GDP': 1.74,
    }

    # --- Contoh 2: Siswa Berisiko Rendah ---
    siswa_risiko_rendah = {
        'Marital_status': 1,
        'Application_mode': 1,
        'Application_order': 1,
        'Course': 33,
        'Daytime_evening_attendance': 1,
        'Previous_qualification': 1,
        'Previous_qualification_grade': 160.0,
        'Nacionality': 1,
        'Mothers_qualification': 4,
        'Fathers_qualification': 4,
        'Mothers_occupation': 2,
        'Fathers_occupation': 2,
        'Admission_grade': 155.0,
        'Displaced': 0,
        'Educational_special_needs': 0,
        'Debtor': 0,
        'Tuition_fees_up_to_date': 1,
        'Gender': 0,
        'Scholarship_holder': 1,
        'Age_at_enrollment': 18,
        'International': 0,
        'Curricular_units_1st_sem_credited': 0,
        'Curricular_units_1st_sem_enrolled': 6,
        'Curricular_units_1st_sem_evaluations': 8,
        'Curricular_units_1st_sem_approved': 6,
        'Curricular_units_1st_sem_grade': 14.5,
        'Curricular_units_1st_sem_without_evaluations': 0,
        'Curricular_units_2nd_sem_credited': 0,
        'Curricular_units_2nd_sem_enrolled': 6,
        'Curricular_units_2nd_sem_evaluations': 8,
        'Curricular_units_2nd_sem_approved': 6,
        'Curricular_units_2nd_sem_grade': 15.0,
        'Curricular_units_2nd_sem_without_evaluations': 0,
        'Unemployment_rate': 9.4,
        'Inflation_rate': -0.3,
        'GDP': 0.79,
    }

    print("=" * 60)
    print("SISTEM PREDIKSI DROPOUT SISWA - Jaya Jaya Institut")
    print("=" * 60)

    for nama, data in [("Siswa A (Risiko Tinggi)", siswa_risiko_tinggi),
                        ("Siswa B (Risiko Rendah)", siswa_risiko_rendah)]:
        print(f"\n[{nama}]")
        try:
            result = predict(data)
            print(f"  Prediksi        : {result['label']}")
            print(f"  Tingkat Risiko  : {result['risk_level']}")
            print(f"  Prob. Dropout   : {result['probability_dropout']:.2%}")
            print(f"  Prob. Graduate  : {result['probability_graduate']:.2%}")
        except FileNotFoundError as e:
            print(f"  ERROR: {e}")

    print("\n" + "=" * 60)
    print("Untuk menggunakan model, pastikan notebook.ipynb sudah dijalankan")
    print("sehingga file model/best_model.pkl tersedia.")
    print("=" * 60)
