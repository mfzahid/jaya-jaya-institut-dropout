# Proyek Akhir: Menyelesaikan Permasalahan Institusi Pendidikan Jaya Jaya Institut

## Business Understanding

Jaya Jaya Institut merupakan institusi pendidikan tinggi yang telah berdiri sejak tahun 2000 dan memiliki ribuan mahasiswa aktif di berbagai program studi. Meskipun telah mencetak banyak lulusan berkualitas, institusi menghadapi tantangan serius berupa tingginya angka **dropout** yang mencapai lebih dari 30% dari total mahasiswa yang memiliki status akhir (tidak termasuk yang masih aktif/enrolled).

Angka dropout yang tinggi berdampak langsung pada:
- **Reputasi dan akreditasi institusi** yang bergantung pada tingkat kelulusan mahasiswa
- **Kerugian finansial** akibat kehilangan pendapatan uang kuliah dan biaya rekrutmen mahasiswa baru yang terus meningkat
- **Dampak sosial** bagi mahasiswa yang dropout, termasuk hilangnya peluang karir dan beban finansial yang sudah dikeluarkan

Pihak yang paling terdampak adalah **tim akademik dan konselor institusi** yang selama ini tidak memiliki alat bantu untuk mengidentifikasi mahasiswa berisiko secara proaktif. Tanpa sistem deteksi dini, intervensi baru dilakukan setelah mahasiswa sudah mengajukan pengunduran diri, sehingga terlambat untuk dicegah.

Dengan memanfaatkan data historis mahasiswa (performa akademik, faktor finansial, dan demografi), model machine learning dapat digunakan untuk **memprediksi risiko dropout sejak semester awal**, memungkinkan tim akademik memberikan bimbingan khusus kepada mahasiswa berisiko sebelum mereka benar-benar memutuskan untuk keluar.

## Permasalahan Bisnis

1. **Dropout rate > 30% tanpa mekanisme pencegahan dini**: Dari data historis, lebih dari sepertiga mahasiswa yang memiliki status akhir tidak berhasil menyelesaikan studi. Tim akademik tidak mengetahui siapa yang berisiko hingga mahasiswa tersebut sudah memutuskan dropout, sehingga tidak ada kesempatan untuk intervensi.

2. **Tidak ada pemahaman berbasis data tentang faktor penyebab dropout**: Konselor dan manajemen akademik belum mengetahui faktor-faktor mana (akademik, finansial, atau demografis) yang paling berkontribusi terhadap dropout, sehingga program bantuan yang ada bersifat umum dan tidak tepat sasaran.

3. **Alokasi sumber daya bimbingan yang tidak efisien**: Karena semua mahasiswa diperlakukan sama tanpa segmentasi risiko, sumber daya bimbingan dan beasiswa tidak tersalurkan secara optimal kepada mahasiswa yang paling membutuhkan.

Solusi data science yang dikembangkan bertujuan menjawab pertanyaan bisnis berikut:
- Faktor apa yang paling berpengaruh terhadap dropout mahasiswa Jaya Jaya Institut?
- Mahasiswa dengan profil seperti apa yang berisiko tinggi untuk dropout?
- Bagaimana institusi dapat mengalokasikan intervensi (bimbingan, beasiswa, keringanan biaya) secara lebih tepat sasaran?

## Cakupan Proyek

1. **Exploratory Data Analysis (EDA)**: Analisis distribusi, pola, dan hubungan antar variabel untuk memahami karakteristik mahasiswa dropout vs graduate
2. **Identifikasi Faktor Dropout**: Analisis statistik dan feature importance untuk menemukan faktor-faktor utama penyebab dropout
3. **Pembangunan Model Prediktif**: Membangun, membandingkan, dan mengevaluasi 3 model machine learning (Logistic Regression, Random Forest, Gradient Boosting) untuk prediksi risiko dropout
4. **Business Dashboard Interaktif**: Membuat dashboard dengan fitur filter dinamis (berdasarkan gender, status beasiswa, status pembayaran, kelompok usia) untuk membantu tim akademik memonitor dan mengeksplorasi data mahasiswa
5. **Prototype Streamlit**: Aplikasi web yang memungkinkan konselor memasukkan data mahasiswa baru dan mendapatkan prediksi risiko dropout secara real-time beserta rekomendasi intervensi

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

Dashboard menampilkan:
- **Overview Panel**: Dropout rate keseluruhan, jumlah siswa aktif, dropout, dan lulus
- **Faktor Akademik**: Performa kurikuler semester 1 dan 2 vs status dropout
- **Faktor Demografis**: Distribusi dropout berdasarkan usia, gender, status pernikahan
- **Faktor Finansial**: Hubungan antara status pembayaran uang kuliah, beasiswa, dan debtor dengan dropout
- **Faktor Sosial-Ekonomi**: Pengaruh tingkat pengangguran, inflasi, dan GDP terhadap dropout
- **Faktor Keluarga**: Kualifikasi dan pekerjaan orang tua vs dropout

## Menjalankan Prototype

Prototype machine learning telah di-deploy pada Streamlit Community Cloud dan dapat diakses melalui:

**Link Prototype**: [https://jaya-jaya-institut-dropout-gowi9aotfsesjwmq9j3pnv.streamlit.app](https://jaya-jaya-institut-dropout-gowi9aotfsesjwmq9j3pnv.streamlit.app)

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
