import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ============================================================
# Page Config
# ============================================================
st.set_page_config(
    page_title="Jaya Jaya Institut - Student Dropout Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Load Model
# ============================================================
def _train_and_save(model_path: str):
    """Train model dari scratch jika pkl tidak bisa diload."""
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.pipeline import Pipeline
    from imblearn.over_sampling import SMOTE

    DATA_URL = (
        'https://raw.githubusercontent.com/dicodingacademy/dicoding_dataset'
        '/main/students_performance/data.csv'
    )
    df = pd.read_csv(DATA_URL, sep=';').dropna(subset=['Status'])
    df = df[df['Status'].isin(['Dropout', 'Graduate'])].copy()
    df['Target'] = (df['Status'] == 'Dropout').astype(int)

    X = df.drop(columns=['Status', 'Target'])
    y = df['Target']

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train_res, y_train_res = SMOTE(random_state=42).fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train_res)

    model = GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1,
        max_depth=4, subsample=0.8, random_state=42
    )
    model.fit(X_scaled, y_train_res)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(Pipeline([('scaler', scaler), ('model', model)]), model_path)
    return joblib.load(model_path)


@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'model', 'best_model.pkl')
    try:
        return joblib.load(model_path)
    except Exception:
        with st.spinner("Melatih model untuk pertama kali, harap tunggu (~1 menit)..."):
            return _train_and_save(model_path)


model = load_model()

# ============================================================
# Helper
# ============================================================
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


def predict_dropout(data: dict) -> dict:
    df = pd.DataFrame([data])[FEATURE_COLUMNS].fillna(0)
    prediction = int(model.predict(df)[0])
    probabilities = model.predict_proba(df)[0]
    prob_dropout = float(probabilities[1])

    if prob_dropout < 0.3:
        risk_level, risk_color = "Rendah", "green"
    elif prob_dropout < 0.6:
        risk_level, risk_color = "Sedang", "orange"
    else:
        risk_level, risk_color = "Tinggi", "red"

    return {
        'prediction': prediction,
        'label': 'Dropout' if prediction == 1 else 'Graduate',
        'probability_graduate': float(probabilities[0]),
        'probability_dropout': prob_dropout,
        'risk_level': risk_level,
        'risk_color': risk_color,
    }


# ============================================================
# Header
# ============================================================
st.title("🎓 Jaya Jaya Institut")
st.subheader("Sistem Prediksi Risiko Dropout Siswa")
st.markdown(
    "Masukkan data siswa untuk memprediksi kemungkinan dropout "
    "dan dapatkan rekomendasi intervensi dini."
)
st.divider()

# ============================================================
# Sidebar - Info
# ============================================================
with st.sidebar:
    st.image("https://via.placeholder.com/150x60?text=Jaya+Jaya+Institut", width=200)
    st.markdown("### Tentang Aplikasi")
    st.markdown(
        "Aplikasi ini menggunakan model **Gradient Boosting** yang dilatih "
        "pada data historis siswa untuk memprediksi risiko dropout."
    )
    st.markdown("**Interpretasi Risiko:**")
    st.markdown("- 🟢 **Rendah** : Prob. Dropout < 30%")
    st.markdown("- 🟡 **Sedang** : Prob. Dropout 30%-60%")
    st.markdown("- 🔴 **Tinggi** : Prob. Dropout > 60%")
    st.divider()
    st.markdown("Jaya Jaya Institut &copy; 2024")

# ============================================================
# Input Form
# ============================================================
tab1, tab2 = st.tabs(["📋 Input Data Siswa", "📊 Prediksi Batch (CSV)"])

with tab1:
    st.markdown("### Data Pribadi & Pendaftaran")
    col1, col2, col3 = st.columns(3)

    with col1:
        marital_status = st.selectbox(
            "Status Pernikahan",
            options=[1, 2, 3, 4, 5, 6],
            format_func=lambda x: {
                1: "Lajang", 2: "Menikah", 3: "Duda/Janda",
                4: "Cerai", 5: "Kumpul Kebo", 6: "Pisah Resmi"
            }[x],
        )
        age = st.number_input("Usia Saat Mendaftar", min_value=17, max_value=70, value=20)
        gender = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Perempuan" if x == 0 else "Laki-laki")

    with col2:
        application_mode = st.number_input("Mode Pendaftaran", min_value=1, max_value=57, value=1)
        application_order = st.number_input("Urutan Pilihan", min_value=0, max_value=9, value=1)
        course = st.number_input("Kode Program Studi", min_value=1, max_value=9999, value=33)

    with col3:
        attendance = st.selectbox(
            "Waktu Kuliah", options=[1, 0],
            format_func=lambda x: "Pagi/Siang" if x == 1 else "Malam"
        )
        prev_qualification = st.number_input("Kualifikasi Sebelumnya", min_value=1, max_value=43, value=1)
        prev_qualification_grade = st.number_input("Nilai Kualifikasi Sebelumnya", min_value=0.0, max_value=200.0, value=130.0)

    st.markdown("### Faktor Finansial & Sosial")
    col4, col5, col6 = st.columns(3)

    with col4:
        debtor = st.selectbox("Status Debtor (Hutang)", options=[0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")
        tuition_up_to_date = st.selectbox(
            "Uang Kuliah Tepat Waktu", options=[1, 0],
            format_func=lambda x: "Ya" if x == 1 else "Tidak"
        )
        scholarship = st.selectbox("Penerima Beasiswa", options=[0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")

    with col5:
        displaced = st.selectbox("Displaced (Pengungsi)", options=[0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")
        educational_special_needs = st.selectbox("Kebutuhan Khusus", options=[0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")
        international = st.selectbox("Mahasiswa Internasional", options=[0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")

    with col6:
        nationality = st.number_input("Kode Nasionalitas", min_value=1, max_value=109, value=1)
        admission_grade = st.number_input("Nilai Penerimaan", min_value=0.0, max_value=200.0, value=130.0)

    st.markdown("### Latar Belakang Keluarga")
    col7, col8 = st.columns(2)
    with col7:
        mothers_qual = st.number_input("Kualifikasi Ibu", min_value=1, max_value=44, value=1)
        fathers_qual = st.number_input("Kualifikasi Ayah", min_value=1, max_value=44, value=1)
    with col8:
        mothers_occ = st.number_input("Pekerjaan Ibu", min_value=0, max_value=195, value=5)
        fathers_occ = st.number_input("Pekerjaan Ayah", min_value=0, max_value=195, value=5)

    st.markdown("### Performa Akademik Semester 1")
    col9, col10, col11 = st.columns(3)
    with col9:
        cu1_credited = st.number_input("SKS Diakui (Sem 1)", min_value=0, max_value=20, value=0)
        cu1_enrolled = st.number_input("SKS Diambil (Sem 1)", min_value=0, max_value=26, value=6)
    with col10:
        cu1_evaluations = st.number_input("Evaluasi (Sem 1)", min_value=0, max_value=45, value=6)
        cu1_approved = st.number_input("SKS Lulus (Sem 1)", min_value=0, max_value=26, value=5)
    with col11:
        cu1_grade = st.number_input("Nilai Rata-rata (Sem 1)", min_value=0.0, max_value=20.0, value=12.0)
        cu1_no_eval = st.number_input("Tanpa Evaluasi (Sem 1)", min_value=0, max_value=12, value=0)

    st.markdown("### Performa Akademik Semester 2")
    col12, col13, col14 = st.columns(3)
    with col12:
        cu2_credited = st.number_input("SKS Diakui (Sem 2)", min_value=0, max_value=20, value=0)
        cu2_enrolled = st.number_input("SKS Diambil (Sem 2)", min_value=0, max_value=23, value=6)
    with col13:
        cu2_evaluations = st.number_input("Evaluasi (Sem 2)", min_value=0, max_value=45, value=6)
        cu2_approved = st.number_input("SKS Lulus (Sem 2)", min_value=0, max_value=20, value=5)
    with col14:
        cu2_grade = st.number_input("Nilai Rata-rata (Sem 2)", min_value=0.0, max_value=20.0, value=12.0)
        cu2_no_eval = st.number_input("Tanpa Evaluasi (Sem 2)", min_value=0, max_value=12, value=0)

    st.markdown("### Indikator Makroekonomi")
    col15, col16, col17 = st.columns(3)
    with col15:
        unemployment_rate = st.number_input("Tingkat Pengangguran (%)", min_value=0.0, max_value=25.0, value=10.8)
    with col16:
        inflation_rate = st.number_input("Tingkat Inflasi (%)", min_value=-5.0, max_value=10.0, value=1.4)
    with col17:
        gdp = st.number_input("GDP", min_value=-5.0, max_value=5.0, value=1.74)

    st.divider()
    predict_btn = st.button("🔍 Prediksi Risiko Dropout", type="primary", use_container_width=True)

    if predict_btn:
        student_data = {
            'Marital_status': marital_status,
            'Application_mode': application_mode,
            'Application_order': application_order,
            'Course': course,
            'Daytime_evening_attendance': attendance,
            'Previous_qualification': prev_qualification,
            'Previous_qualification_grade': prev_qualification_grade,
            'Nacionality': nationality,
            'Mothers_qualification': mothers_qual,
            'Fathers_qualification': fathers_qual,
            'Mothers_occupation': mothers_occ,
            'Fathers_occupation': fathers_occ,
            'Admission_grade': admission_grade,
            'Displaced': displaced,
            'Educational_special_needs': educational_special_needs,
            'Debtor': debtor,
            'Tuition_fees_up_to_date': tuition_up_to_date,
            'Gender': gender,
            'Scholarship_holder': scholarship,
            'Age_at_enrollment': age,
            'International': international,
            'Curricular_units_1st_sem_credited': cu1_credited,
            'Curricular_units_1st_sem_enrolled': cu1_enrolled,
            'Curricular_units_1st_sem_evaluations': cu1_evaluations,
            'Curricular_units_1st_sem_approved': cu1_approved,
            'Curricular_units_1st_sem_grade': cu1_grade,
            'Curricular_units_1st_sem_without_evaluations': cu1_no_eval,
            'Curricular_units_2nd_sem_credited': cu2_credited,
            'Curricular_units_2nd_sem_enrolled': cu2_enrolled,
            'Curricular_units_2nd_sem_evaluations': cu2_evaluations,
            'Curricular_units_2nd_sem_approved': cu2_approved,
            'Curricular_units_2nd_sem_grade': cu2_grade,
            'Curricular_units_2nd_sem_without_evaluations': cu2_no_eval,
            'Unemployment_rate': unemployment_rate,
            'Inflation_rate': inflation_rate,
            'GDP': gdp,
        }

        with st.spinner("Menganalisis data siswa..."):
            result = predict_dropout(student_data)

        st.divider()
        st.markdown("## Hasil Prediksi")

        col_res1, col_res2, col_res3 = st.columns(3)
        with col_res1:
            if result['prediction'] == 1:
                st.error(f"### Prediksi: {result['label']}")
            else:
                st.success(f"### Prediksi: {result['label']}")
        with col_res2:
            st.metric("Probabilitas Dropout", f"{result['probability_dropout']:.1%}")
        with col_res3:
            st.metric("Probabilitas Graduate", f"{result['probability_graduate']:.1%}")

        risk_emoji = {"Rendah": "🟢", "Sedang": "🟡", "Tinggi": "🔴"}
        st.markdown(f"### Tingkat Risiko: {risk_emoji[result['risk_level']]} **{result['risk_level']}**")

        if result['risk_level'] == "Tinggi":
            st.warning(
                "**Rekomendasi**: Siswa ini berisiko tinggi dropout. "
                "Segera hubungi siswa untuk konseling akademik dan finansial. "
                "Periksa status pembayaran uang kuliah dan tawarkan opsi beasiswa atau cicilan."
            )
        elif result['risk_level'] == "Sedang":
            st.info(
                "**Rekomendasi**: Siswa ini berisiko sedang. "
                "Monitor performa akademik semester berikutnya secara berkala. "
                "Tawarkan program mentoring dan bimbingan belajar."
            )
        else:
            st.success(
                "**Rekomendasi**: Siswa ini berisiko rendah. "
                "Pertahankan dukungan akademik dan dorong keterlibatan dalam kegiatan kampus."
            )

# ============================================================
# Tab 2: Batch Prediction
# ============================================================
with tab2:
    st.markdown("### Prediksi Batch dari File CSV")
    st.markdown(
        "Upload file CSV dengan kolom yang sama seperti dataset siswa. "
        "Kolom `Status` tidak diperlukan."
    )

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file, sep=None, engine='python')
            st.dataframe(df_upload.head(), use_container_width=True)

            if st.button("Jalankan Prediksi Batch", type="primary"):
                with st.spinner("Memproses data..."):
                    cols = [c for c in FEATURE_COLUMNS if c in df_upload.columns]
                    df_input = df_upload[cols].fillna(0)

                    predictions = model.predict(df_input)
                    probabilities = model.predict_proba(df_input)

                    df_result = df_upload.copy()
                    df_result['Prediksi'] = ['Dropout' if p == 1 else 'Graduate' for p in predictions]
                    df_result['Prob_Dropout'] = probabilities[:, 1].round(4)
                    df_result['Prob_Graduate'] = probabilities[:, 0].round(4)
                    df_result['Risk_Level'] = df_result['Prob_Dropout'].apply(
                        lambda x: 'Tinggi' if x >= 0.6 else ('Sedang' if x >= 0.3 else 'Rendah')
                    )

                st.success(f"Prediksi selesai untuk {len(df_result)} siswa.")

                col_m1, col_m2, col_m3 = st.columns(3)
                with col_m1:
                    st.metric("Total Siswa", len(df_result))
                with col_m2:
                    n_dropout = (df_result['Prediksi'] == 'Dropout').sum()
                    st.metric("Prediksi Dropout", n_dropout)
                with col_m3:
                    st.metric("Risiko Tinggi", (df_result['Risk_Level'] == 'Tinggi').sum())

                st.dataframe(df_result, use_container_width=True)

                csv_out = df_result.to_csv(index=False)
                st.download_button(
                    "Download Hasil Prediksi",
                    data=csv_out,
                    file_name="hasil_prediksi_dropout.csv",
                    mime="text/csv",
                )
        except Exception as e:
            st.error(f"Error membaca file: {e}")
