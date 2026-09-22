"""
Script untuk membersihkan data siswa dan mengekspor CSV yang siap digunakan
untuk Business Dashboard (Metabase / Looker Studio).
"""
import pandas as pd

DATA_URL = (
    'https://raw.githubusercontent.com/dicodingacademy/dicoding_dataset'
    '/main/students_performance/data.csv'
)

df = pd.read_csv(DATA_URL, sep=';')

# Drop baris dengan Status null (target variable tidak boleh di-impute)
df = df.dropna(subset=['Status'])

# Fill null pada kolom lainnya
for col in df.columns:
    if df[col].isnull().any():
        if df[col].dtype == 'object':
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)

# Tambah kolom label biner untuk kemudahan filter di dashboard
df['Is_Dropout'] = (df['Status'] == 'Dropout').astype(int)

# Tambah kolom kategori usia
df['Age_Group'] = pd.cut(
    df['Age_at_enrollment'],
    bins=[0, 20, 25, 30, 40, 100],
    labels=['<=20', '21-25', '26-30', '31-40', '>40']
)

# Tambah kolom label gender
df['Gender_Label'] = df['Gender'].map({1: 'Male', 0: 'Female'})

# Tambah kolom label scholarship
df['Scholarship_Label'] = df['Scholarship_holder'].map({1: 'Yes', 0: 'No'})

# Tambah kolom label debtor
df['Debtor_Label'] = df['Debtor'].map({1: 'Yes', 0: 'No'})

# Tambah kolom label tuition fees
df['Tuition_Label'] = df['Tuition_fees_up_to_date'].map({1: 'Up to Date', 0: 'Not Up to Date'})

df.to_csv('students_cleaned.csv', index=False)
print(f'Done! {len(df)} rows saved to students_cleaned.csv')
print(f'Status distribution: {df["Status"].value_counts().to_dict()}')
print(f'Dropout rate: {df["Is_Dropout"].mean() * 100:.2f}%')
