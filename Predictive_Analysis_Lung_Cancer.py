# Import Library
import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from scipy.stats import randint

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import kagglehub

# Import Dataset
# Download latest version
path = kagglehub.dataset_download("amankumar094/lung-cancer-dataset")
file_path = os.path.join(path, 'dataset_med.csv')
df = pd.read_csv(file_path)
df = df.drop(columns=['id'])

# Memastikan tipe data datetime
df['diagnosis_date'] = pd.to_datetime(df['diagnosis_date'], errors='coerce', format='%Y-%m-%d')
df['end_treatment_date'] = pd.to_datetime(df['end_treatment_date'], errors='coerce', format='%Y-%m-%d')

# Handling outlier
filtered_df = df[(df['diagnosis_date'] < pd.to_datetime('2024-12-31')) & (df['end_treatment_date'] < pd.to_datetime('2024-12-31'))]

df = filtered_df.copy()
print('Ukuran dataset : ', df.shape)

# Membagi data menjadi type data number dan kategorikal
num_cols = df.select_dtypes(include='number').columns
cat_cols = df.select_dtypes(include=['object', 'category']).columns

def boxplot_num_cols(df, num_cols, title):
  """
  Menampilkan visualisasi boxplot pada setiap fitur numerik
  """
  n_num_col = len(num_cols) # Jumlah kolom per baris
  cols = 3
  rows = math.ceil(n_num_col/cols)

  # Membuat plot
  fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
  axes = axes.flatten()

  for i, col in enumerate(num_cols):
      sns.boxplot(y=df[col], ax=axes[i])
      axes[i].set_title(f'Boxplot: {col}', fontsize=10)

  # Hapus subplot kosong jika ada
  for j in range(i + 1, len(axes)):
      fig.delaxes(axes[j])

  fig.suptitle(title, fontsize=16)
  plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Supaya judul utama tidak ketindih
  plt.show()

before_iqr = "Sebelum Handling Outlier"
boxplot_num_cols(df, num_cols, before_iqr)

# Handling outlier using IQR
Q1 = df['age'].quantile(0.25)
Q3 = df['age'].quantile(0.75)
IQR = Q3 - Q1

filter_outiers = ~((df['age'] < (Q1 - 1.5 * IQR)) | (df['age'] > (Q3 + 1.5 * IQR)))
df = df[filter_outiers].reset_index(drop=True)

print('Ukuran dataset setelah handling outlier', df.shape)

after_IQR = "Setelah Handling Outlier"
boxplot_num_cols(df, num_cols, after_IQR)

# Univariate Data Analysise

for col in cat_cols:
    count = df[col].value_counts()
    percent = 100 * df[col].value_counts(normalize=True)
    data = pd.DataFrame({'Jumlah Sampel': count, 'Percent': percent.round(1)})

    print(f"\nDistribusi fitur: {col}")
    print(data)

'''Visualisasi diagram barang untuk semua nilai kategorikal'''
n = len(cat_cols)
n_cols = 3
n_rows = (n + n_cols - 1) // n_cols

plt.figure(figsize=(5 * n_cols, 5 * n_rows))

for i, col in enumerate(cat_cols, 1):
    plt.subplot(n_rows, n_cols, i)
    sns.countplot(data=df, x=col, hue=col, palette='viridis', legend=False)
    plt.title(f'Grafik dari Jumlah {col}')
    plt.xticks(rotation=45)
    plt.xlabel('')
    plt.ylabel('count')

plt.tight_layout()
plt.show()
'''Dari kedua visualisasi univariate di atas, dapat diketahui bahwa:
1. Semua fitur kategorikal memiliki distribusi yang seimbang pada masing-masing nilai uniknya.
2. Pada fitur `age` pasien laki-laki dan perempuan sama banyaknya.
3. Persebaran negara pada fitur `country` memiliki distribusi yang seimbang atau jumlah pada setiap nilainya sama.
4. Pada fitur `cancer_stage`, Sebagian besar pasien kanker paru-paru terdiagnosis pada Stadium IV, diikuti oleh Stadium III, Stadium II, dan paling rendah pada Stadium I.
5. Pasien dengan riwayat anggota keluarga menderita kanker paru-paru memiliki potensi sama besarnya dengan pasien yang keluarganya tidak memiliki riwayat kanker paru-paru.
6. Riwayat kanker paru-paru lebih tinggi pada perokok pasif dan paling rendah pada perokok aktif.
7. Penanganan paling banyak dilakukan pada pasien kanker paru-paru dengan operasi dan kemoterapi, sedangkan yang paling sedikit menggunakan radiasi.
'''
# Univariate Data Numerik
'''Histogram'''
df.hist(bins=50, figsize=(20,15))
plt.show()

'''
Dari grafik di atas, dapat diketahui bahwa:
1. Disrtibusi pada fitur `age` merupakan distribusi normal dan menunjukkan kenaikan nilai di beberapa angka.
2. Distribusi pada fitur `diagnosis_date` menyebar dan mengalami penurunan pasien setelah tahun 2022. Bisa dikatakan persebaran sedikit miring ke kiri atau left-skewd.
3. Persebaran pada fitur `bmi` distribusinya merata diantara 18-45 dan menunjukkan variasi yang luas.
4. Pada fitur `colesterol_level` distribusi memiliki dua puncak atau bimodal setelah nilai sekitar 240. Bisa dikategorikan terdapat kelompok dengan risiko kolesterol tinggi.
5. Pada fitur `end_treatment_date` distribusinya mirip `diagnosis_date`, pasien mengalami kenaikan nilai setelah tahun 2015.
5. Pada fitur `hypertension`, `asthma`, `cirrhosis`, `other_cancer` dan `survived` merupakan variabel biner dengan nilai 0 (tidak memiliki kondisi tersebut) dan 1 (memiliki kondisi tersebut). Dapat disimpulkan bahwa:
- `hypertension`, lebih banyak pasien dengan kondisi tekanan darah tinggi (nilai 1 paling tinggi).
- `asthma`, sebagian besar pasien tidak memiliki penyakit asma.
- `cirrhosis`, hanya sebagian kecil pasien memiliki kondisi ini.
- `other_cancer`, hanya sebagian kecil pasien memiliki kondisi ini.
- `survived`, mayoritas pasien tidak bertahan hidup (nilai 0 paling tinggi).
'''

# Distribusi fitur survial yang merupakan target
survival_counts = df['survived'].value_counts().sort_index()

# Label bisa disesuaikan kalau ingin lebih informatif
labels = ['Tidak Bertahan (0)', 'Bertahan (1)']

# Pie chart
plt.figure(figsize=(6, 6))
# colors = plt.cm.viridis(np.linspace(0, 1, len(survival_counts)))
plt.pie(survival_counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("Set2"))
plt.title('Distribusi Fitur Survival')
plt.axis('equal')
plt.show()

# Visualisasi multivariate data
# Correlation matrix
plt.figure(figsize=(12,6))
sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Data Preparation

'''
Fungsi pairplot di atas menjukkan relasi pasangan dalam dataset. Diketahui bahwa:
1. Distribusi pada fitur `age` merupakan distribusi normal.
2. Korelasi fitur `colesterol_level` dengan `bmi` menunjukkan miring ke kanan atau right-skewd menunjukkan bahwa korelasi yang positif terjadi di antara kedua fitur tersebut.
3. Selain dari dari ketiga fitur di atas, merupakan fitur kategorikal yang diubah menjadi numerik sehingga korelasinya merupakan garis lurus.
4. Dapat dipastikan, karakteristik dataset merupakan non-linear dan tersebar.

4. Data Preparation
Feature Engineering
Menambahkan beberapa fitur yang mungkin relevan agar model dapat memberikan klasifikasi dengan performa yang baik. Nerikut fitur yang akan ditambahkan:
1. `survival_months`, merupakan selisih dari `diagnosis_date` dengan `end_treatment_date` yang disajikan dalam bentuk bulan.
2. `survival_group`, hasil binning dari `survival_months` yang disajikan dalam bentuk interval 2 tahun.
3. `age_group`, hasil binning dari `age` dengan membagi menjadi rentang puluhan, misalnya usia 20an, 30an, dan seterusnya.
4. `cholesterol_group`, hasil binning dari `cholesterol_level` yang dibagi menjadi Normal jika di bawah angka 200 dan High jika kadar kolesterol di atas 200.
5. `bmi_group`, hasil binning dari `bmi` yang dibagi menjadi Underweight (<18), Normal (18,5-24,9), dan Overwight (>25).
'''

# Feature engineering dengan teknik bining pada fitur age, cholesterol_level, dan bmi

bins_age = [20, 30, 40, 50, 60, 70, 80, 90]
labels_age = ['20an', '30an', '40an', '50an', '60an', '70an', '80an']

bins_survival = [0, 24, 48, 72, 96, 120, 144]
survival_labels = ['0-2 Years', '2-4 Years', '4-6 Years', '6-8 Years', '8-10 Years', '10-12 Years']
duration_of_treatment = df['end_treatment_date'] - df['diagnosis_date']

bins_colesterol = [0, 200, 300]
labels_colesterol = ['Normal', 'High']

bins_bmi = [0, 18.5, 24.9, 50]
labels_bmi = ['Underweight', 'Normal', 'Overweight']

df['survival_months'] = round(duration_of_treatment.dt.days / 30)
df['survival_group'] = pd.cut(df['survival_months'], bins=bins_survival, labels=survival_labels, right=False)
df['age_group'] = pd.cut(df['age'], bins=bins_age, labels=labels_age, right=False)
df['cholesterol_group'] = pd.cut(df['cholesterol_level'], bins=bins_colesterol, labels=labels_colesterol, right=False)
df['bmi_group'] = pd.cut(df['bmi'], bins=bins_bmi, labels=labels_bmi, right=False)

'''
Setelah melakukan feature engineering, hampir semua fitur kategori terhadap `survival_months` memiliki 
distribusi yang sama, kecuali pada fitur `cancer_stage`. Semakin rendah stadium pada pasien, maka perawatannya 
lebih lama. Hal ini memberikan interpretasi bahwa kanker dengan stadium semakin tinggi, 
semakin kecil harapan untuk selamat sesuai dengan distribusi pada fitur `survival`.
'''

# Split data

X = df.drop(columns=['survived', 'diagnosis_date', 'end_treatment_date'])
y = df['survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# Label Encoding
'''
Mengubah nilai data kategorikal menjadi representasi numerik dilakukan agar komputer dapat lebih mudah mengenali fitur tersebut. 
Proses label encoding dilakukan secara manual pada fitur supaya lebih mudah dipantau. Proses ini dilakukan setelah proses spliting 
dataset (splitting) dilakukan untuk mencegah terjadinya kebocoran data (data leakage) pada model.
'''
# Kembali menginisiasi kolom dengan tipe numerikal untuk standar scaler dan objek untuk label encoder
num_cols = ['age', 'bmi', 'cholesterol_level', 'survival_months']
cat_cols = ['gender', 'cancer_stage','survival_group', 'country', 'family_history', 'smoking_status', 'treatment_type', 
            'age_group', 'cholesterol_group', 'bmi_group']

def labelEncoding(df,columns):
    for col in columns:
        mapping = {val: i for i, val in enumerate(df[col].unique(), 1)}
        df[col] = df[col].map(mapping)
    return df[columns]

X_train[cat_cols] = labelEncoding(X_train,cat_cols)
X_test[cat_cols] = labelEncoding(X_test,cat_cols)

# Menampilkan data train dan test
print(f"Training Shape: {X_train.shape}, {y_train.shape}")
print(f"Test Shape: {X_test.shape}, {y_test.shape}")

# Normalisasi
'''
Normalisasi data dilakukan mennggunakan standard scaler pada fitur numerik terutama bagi data yang memiliki selisih yang tinggi. 
Nilai numerik yang dilakukan normalisasi, yaitu `age`, `bmi`, `cholesterol_level`, dan `survival_months`.
'''
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# Feature Selection
'''
Tidak semua fitur akan digunakan pada proses pembangunan model, mengingat terdapat 18 fitur pada data. Maka akan dilakukan seleksi 
fitur menggunakan teknik Lasso dan mempertimbangkan matriks korelasi (correlation matrix).
'''
plt.figure(figsize=(16,10))
sns.heatmap(X_train.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix After Cleaning Dataset')
plt.show()

'''
Dari korelasi matriks di atas, terdapat beberapa fitur yang saling berkorelasi dengan nilai yang cukup tinggi, yaitu:
1. Fitur `age` dengan `age_group` memiliki korelasi negatif.
2. `bmi`, `bmi_group`, `cholesterol_level`, dan `colesterol_group` memiliki korelasi dengan nilai yang cukup signifikan dan dapat mempengaruhi model.
3. `survival_month` dan `survival_group` memiliki korelasi yang positif.
4. Maka dari itu, ke delapan fitur tersebut dapat dijadikan bahan pertimbangan untuk dimasukkan ke dalam model.

Banyaknya jumlah fitur, distribusi yang hampir merata pada sebagian besar fitur, serta korelasi antar fitur yang rendah dapat menyebabkan model membuat prediksi yang terlalu umum (general). Oleh karena itu, dilakukan proses pemilihan fitur (feature selection) untuk meningkatkan kinerja model menggunakan teknik Lasso.

**Teknik Lasso** (Least Absolute Shrinkage and Selection Operator) merupakan salah satu teknik regresi dalam statistika dan machine learning untuk pilihan fitur. Lasso menambahkan penalti berupa nilai absolut ke dalam fungsi loss, yang disebut dengan L1 regularization. Bekerja dengan membuat beberapa koefisien regresi menjadi nol sehingga fitur yang tidak penting dihilangkan dari model.

Kelebihan dari Lasso:
1. Dapat melakukan seleksi fitur pada model.
2. Membantu menangani overfitting.
3. Cocok untuk dataset dengan banyak fitur, terutama jika beberapa fitur tidak penting digunakan.
'''

# Pakai Lasso untuk mengecilkan koefisien fitur tidak penting ke nol
lasso = Lasso(alpha=0.001)
lasso.fit(X_train, y_train)

# Lihat fitur yang dipilih
selected_features = X.columns[lasso.coef_ != 0]
print("Selected Features:", selected_features)

selected_features_cols = selected_features.tolist() + ['survival_months', 'bmi', 'cholesterol_group']

# Mengambil data X train dan test yang sudah dipisahkan sebelumnya.
X_train_selected = X_train[selected_features_cols]
X_test_selected = X_test[selected_features_cols]

'''
Diputuskan, total 6 fitur yang akan digunakan dalam proses pembangunan model ini. Fitur-fitur yang dipilih, yaitu `country`, 
`cancer_stage`, `age_group`, `survival_months`, `bmi`, dan `cholesterol_group`.
'''

# Building Model
# Decision Tree
dt_model = DecisionTreeClassifier(random_state=42, max_depth=10)
dt_model.fit(X_train_selected, y_train)

# Random Forest
rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
rf_model.fit(X_train_selected, y_train)

# KNN
knn_model = KNeighborsClassifier(n_neighbors=5, metric='minkowski')
knn_model.fit(X_train_selected, y_train)

# XGBoost
gb_model = GradientBoostingClassifier(n_estimators=50, random_state=42)
gb_model.fit(X_train_selected, y_train)

# Evaluasi model
models = {'Decission Tree':dt_model, 'Random Forest':rf_model, 'KNN':knn_model, 'Gradient Boosting':gb_model}

for name, model in models.items():
  y_pred = model.predict(X_test_selected)
  accuracy = accuracy_score(y_test, y_pred)
  print(confusion_matrix(y_test, y_pred))
  print(f'\n{name} Accuracy: {accuracy:.4f}')
  print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=1))

def evaluasi_model_train(models, X_train, y_train):
    results = []
    for name, model in models.items():
            y_pred = model.predict(X_train)
            accuracy = accuracy_score(y_train, y_pred)
            precision = precision_score(y_train, y_pred)
            recall = recall_score(y_train, y_pred)
            f1 = f1_score(y_train, y_pred)

            results.append([name, accuracy, precision, recall, f1])

    return pd.DataFrame(results, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

# Panggil fungsi
print('Evalluasi matriks training')
df_evaluasi_train = evaluasi_model_train(models, X_train_selected, y_train)
df_evaluasi_train

def evaluasi_model_test(models, X_test, y_test):
    results = []

    for name, model in models.items():
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        results.append([name, accuracy, precision, recall, f1])

    return pd.DataFrame(results, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

print('Evaluasi matriks testing')
df_evaluasi_test = evaluasi_model_test(models, X_test_selected, y_test)
df_evaluasi_test