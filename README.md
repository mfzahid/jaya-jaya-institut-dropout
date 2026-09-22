# Proyek Akhir: Menyelesaikan Permasalahan Institusi Pendidikan Jaya Jaya Institut

## Business Understanding

Jaya Jaya Institut merupakan salah satu institusi pendidikan perguruan yang telah berdiri sejak tahun 2000. Hingga saat ini ia telah mencetak banyak lulusan dengan reputasi yang sangat baik. Akan tetapi, terdapat banyak juga siswa yang tidak menyelesaikan pendidikannya alias **dropout**.

Jumlah dropout yang tinggi ini tentunya menjadi salah satu masalah yang besar untuk sebuah institusi pendidikan. Oleh karena itu, Jaya Jaya Institut ingin mendeteksi secepat mungkin siswa yang mungkin akan melakukan dropout sehingga dapat diberi bimbingan khusus.

## Permasalahan Bisnis

1. **Tingginya Angka Dropout**: Institusi mengalami angka dropout yang signifikan, berdampak pada reputasi, akreditasi, dan keberlanjutan institusi.
2. **Kurangnya Sistem Deteksi Dini**: Tidak ada mekanisme untuk mengidentifikasi siswa berisiko dropout sebelum mereka benar-benar keluar.
3. **Kurangnya Pemahaman Faktor Penyebab**: Belum ada analisis mendalam mengenai faktor-faktor utama yang mendorong siswa untuk dropout, sehingga intervensi tidak dapat dilakukan secara tepat sasaran.

## Cakupan Proyek

1. **Exploratory Data Analysis (EDA)**: Analisis menyeluruh terhadap data siswa untuk memahami distribusi, pola, dan hubungan antar variabel terhadap dropout
2. **Identifikasi Faktor Dropout**: Menganalisis faktor-faktor paling berpengaruh terhadap keputusan siswa untuk dropout
3. **Pembangunan Model Prediktif**: Membangun dan mengevaluasi model machine learning untuk memprediksi siswa yang berpotensi dropout
4. **Business Dashboard**: Membuat dashboard interaktif untuk monitoring faktor-faktor dropout
5. **Prototype Streamlit**: Membangun aplikasi web untuk prediksi dropout secara real-time

## Persiapan

**Sumber Data**: [Students Performance Dataset - Dicoding GitHub](https://github.com/dicodingacademy/dicoding_dataset/blob/main/students_performance/README.md)

**Setup Environment**:

- **Python Version**: 3.9+

```bash
# Buat virtual environment
python -m venv venv

# Aktifkan virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Menjalankan Notebook**:

```bash
jupyter notebook notebook.ipynb
```

**Menjalankan Aplikasi Streamlit**:

```bash
streamlit run app.py
```

## Business Dashboard

Dashboard telah dibuat menggunakan **Looker Studio (Google Data Studio)** untuk memvisualisasikan faktor-faktor yang mempengaruhi dropout siswa di Jaya Jaya Institut.

**Link Dashboard**: [Jaya Jaya Institut Student Dropout Dashboard](https://datastudio.google.com/u/0/reporting/125244c1-3ae4-4b65-ab1a-8348e640c903/page/ezX9F)

**Email Metabase**: root@mail.com
**Password Metabase**: root123

Dashboard menampilkan:
- **Overview Panel**: Dropout rate keseluruhan, jumlah siswa aktif, dropout, dan lulus
- **Faktor Akademik**: Performa kurikuler semester 1 dan 2 vs status dropout
- **Faktor Demografis**: Distribusi dropout berdasarkan usia, gender, status pernikahan
- **Faktor Finansial**: Hubungan antara status pembayaran uang kuliah, beasiswa, dan debtor dengan dropout
- **Faktor Sosial-Ekonomi**: Pengaruh tingkat pengangguran, inflasi, dan GDP terhadap dropout
- **Faktor Keluarga**: Kualifikasi dan pekerjaan orang tua vs dropout

## Menjalankan Prototype

Prototype machine learning telah di-deploy pada Streamlit Community Cloud dan dapat diakses melalui:

**Link Prototype**: [https://jaya-jaya-institut-dropout.streamlit.app](https://jaya-jaya-institut-dropout.streamlit.app)

Untuk menjalankan secara lokal:

```bash
# Install dependencies
pip install -r requirements.txt

# Jalankan Streamlit
streamlit run app.py
```

## Conclusion

Berdasarkan analisis data dan model machine learning yang telah dibangun, dapat disimpulkan:

### Temuan Utama

1. **Dropout Rate Aktual**: Jaya Jaya Institut memiliki dropout rate sekitar **32%** dari total siswa (setelah mengecualikan siswa yang masih enrolled), yang merupakan angka cukup tinggi dan perlu segera ditangani.

2. **Faktor-Faktor Utama Penyebab Dropout**:
   - **Performa Akademik Semester 2**: Jumlah mata kuliah yang disetujui (approved) dan nilai (grade) semester 2 adalah prediktor terkuat. Siswa dengan nilai rendah di semester 2 sangat berisiko dropout.
   - **Performa Akademik Semester 1**: Sama halnya dengan semester 2, performa semester 1 yang buruk menjadi tanda awal risiko dropout.
   - **Status Pembayaran Uang Kuliah (Tuition_fees_up_to_date)**: Siswa yang tidak membayar uang kuliah tepat waktu memiliki risiko dropout jauh lebih tinggi.
   - **Usia saat Pendaftaran (Age_at_enrollment)**: Siswa yang mendaftar di usia lebih tua cenderung lebih rentan dropout, kemungkinan karena harus menyeimbangkan pekerjaan dan pendidikan.
   - **Status Beasiswa (Scholarship_holder)**: Penerima beasiswa memiliki dropout rate yang lebih rendah, menunjukkan bahwa dukungan finansial berperan penting.
   - **Status Debtor**: Siswa yang memiliki hutang pada institusi memiliki risiko dropout lebih tinggi.
   - **Kualifikasi Sebelumnya**: Siswa dengan kualifikasi akademik sebelumnya yang lebih rendah cenderung lebih sulit mengikuti perkuliahan.

3. **Performa Model Machine Learning**:
   | Model | Accuracy | ROC-AUC | F1-Score (Dropout) |
   |-------|----------|---------|-------------------|
   | Logistic Regression | ~78% | ~0.84 | ~0.72 |
   | Random Forest | ~85% | ~0.91 | ~0.80 |
   | Gradient Boosting | **~87%** | **~0.93** | **~0.83** |

   Model terbaik yang dipilih adalah **Gradient Boosting Classifier**.

## Rekomendasi Action Items

Berdasarkan temuan analisis, berikut adalah rekomendasi konkret untuk Jaya Jaya Institut:

1. **Program Intervensi Akademik Dini**
   - Identifikasi siswa dengan nilai kurikuler rendah di akhir semester 1 dan segera berikan bimbingan belajar tambahan
   - Buat sistem early warning berbasis model prediktif yang sudah dikembangkan untuk memantau siswa berisiko setiap akhir semester

2. **Program Bantuan Finansial**
   - Perluas program beasiswa untuk siswa berprestasi dari keluarga kurang mampu
   - Buat skema cicilan pembayaran uang kuliah yang lebih fleksibel untuk mengurangi jumlah siswa yang tertunggak
   - Sediakan program kerja paruh waktu di kampus untuk siswa yang membutuhkan dukungan finansial

3. **Dukungan untuk Mahasiswa Usia Lebih Tua**
   - Buat program kuliah kelas malam atau hybrid untuk mahasiswa yang bekerja
   - Sediakan konseling khusus untuk mahasiswa dewasa yang harus menyeimbangkan tanggung jawab keluarga dan pekerjaan

4. **Program Onboarding dan Orientasi**
   - Perkuat program orientasi untuk siswa dengan kualifikasi sebelumnya yang lebih rendah
   - Sediakan kelas remedial/persiapan untuk mata kuliah dasar sebelum semester dimulai

5. **Sistem Monitoring Berkelanjutan**
   - Implementasikan dashboard monitoring yang diperbarui setiap semester untuk melacak siswa berisiko
   - Gunakan prototype Streamlit yang telah dibuat untuk memprediksi dan memantau risiko dropout secara proaktif
   - Lakukan survei kepuasan siswa secara berkala untuk mengidentifikasi masalah sebelum berujung pada dropout

6. **Penguatan Layanan Konseling**
   - Tingkatkan kapasitas layanan konseling akademik dan psikologis
   - Adakan sesi mentoring antara siswa senior dan junior untuk membangun komunitas belajar yang suportif
