# Laporan Proyek Machine Learning - Alya Fauzia Azizah

## Domain Proyek

Kanker paru didefinisikan sebagai pertumbuhan sel kanker yang tidak normal dan tidak terkendali pada jaringan paru dengan penyebab utamanya oleh zat kimia yang memiliki sifat karsinogenik[[1]](#ref1). Asap rokok, merupakan salah satu zat yang memiliki sifat karsinogen yang jika dihirup atau terpapar dalam jangka waktu yang lama akan menyebabkan gangguan pada epitel saluran pernapasan dan berpotensi mengakibatkan sakit dada. Penyebab lainnya bisa ditambah oleh konsumsi alkohol yang berlebihan, kondisi udara yang tercemar, dan gaya hidup yang tidak sehat[[2]](#ref2).

Kejadian pada kanker paru saat ini menjadi penyebab kematian tertinggi akibat kanker dan sebanding dengan banyaknya kejadian. Pada tahun 2022 menurut Global Cancer Observatory (Globocan), dalam lima tahun terakhir (2018-2022) mencapai 12,5% dari total kasus kanker di dunia atau sekitar 2.480 juta pasien. Sedangkan di Indonesia, kasus kanker paru mencapai angka 38.904 kasus atau sekitar 9,5% dari total kasus kanker. Kejadian ini menjadikan kanker paru menjadi kasus tertinggi ke dua setelah kanker payudara[[3]](#ref3).

Lamanya pasien bertahan hidup atau _Median Survival_ pasien kanker paru sekitar 4-5 bulan dengan terapi, pasien dengan _survival rate_ lebih dari 1 tahun tidak lebih dari 10%. Pilihan prioritas terapi merupakan terapi bedah, namun hanya 20% penderita yang dapat menjalankannya. Karena sebagian besar sel kanker penderita yang terdiagnosis sudah tersebar, datang terlambat, atau bahkan sudah stadium lanjut[[4]](#ref4). Salah satu kunci untuk menangani dan mencegah kanker, salah satuya kanker paru dapat dengan deteksi dini. Sulit untuk menyembuhkan pasien kanker yang sudah terdiagnosis, tetapi banyak juga pasien kanker yang dapat diobati secara efektif jika ditemukan pada tahap awal[[5]](#ref5). Pendeteksian dini dapat menggunakan teknologi berbasis Machine Learning yang mampu memberikan hasil dengan cepat. Hasil prediksi ini dapat menjadi bahan pertimbangan tenaga medis dalam proses diagnosis, sehingga proses pengambilan keputusan menjadi lebih efektif dan efisien[[2]](#ref2).

## Business Understanding

### Problem Statements

Berdasarkan latar belakang di atas, dapat diperoleh permasalahan sebagai berikut:
1. Fitur-fitur apa saja yang paling berpengaruh terhadap diagnosis dan tingkat keselamatan pasien kanker paru-paru?
2. Bagaimana hubungan antara keberlangsungan hidup pasien dengan stadium kanker dan jenis perawatan yang diterima?
3. Model machine learning mana yang paling optimal dalam mengklasifikasikan tingkat keselamatan pasien kanker paru-paru?

### Goals

Maka dari itu, diperoleh tujuan sebagai berikut:
1. Mengidentifikasi faktor-faktor yang memengaruhi diagnosis dan risiko kanker paru-paru pada pasien.
2. Menganalisis hubungan antara jenis perawatan yang diterima dengan tingkat keberlangsungan hidup pasien kanker paru-paru.
3. Membangun dan membandingkan beberapa model machine learning untuk mendapatkan model terbaik dalam mengklasifikasikan tingkat risiko kanker paru-paru.


### Solution statements
Untuk menyelesaikan permasalahan dan mencapai tujuan, akan dilakukan solusi:
- Menggunakan empat algoritma klasifikasi dengan karakteristik yang berbeda agar mendapatkan performa dan model yang paling optimal menyesuaikan karakteristik data. 
- Fitur dalam dataset akan banyak dan karakteristiknya beragam, maka dari itu akan digunakan _feature selection_ menggunakan teknik Lasso dan mempertimbangkan matriks korelasi antar fitur.
- Selain menggunakan akurasi, metrik evaluasi akan mempertimbangkan f1-score, recall, dan precision dengan menyocokan terhadap _confussion matriks_.

## Data Understanding

Dataset yang digunakan pada proyek ini merupakan basis data yang diambil dari pasien yang komprehensif, dikhususkan pada individu yang didiagnosis menderita kanker paru-paru. Diambil dari platform [Kaggle](kaggle.com/datasets/amankumar094/lung-cancer-dataset) dengan judul Lung Cancer Dataset, basis data ini dibuat untuk menganalisis berbagai faktor pengobatan yang dapat mempengaruhi diagnosis kanker dan pengobatan. Terdiri dari 890.000 data dan 16 variabel pengamatan yang terdiri dari kondisi medis pasien.


<img src="https://raw.githubusercontent.com/ayalya/PredictiveAnalysis_LungCancer/main/asset/1Datatype.png" align="center"><a></a>

*Gambar 1, Type Data Variable*

### Variabel-variabel pada Lung Cancer Dataset dari Kaggle adalah sebagai berikut:
*Tabel 1, Deskripsi Variabel*
Variabel | Keterangan
----------|----------
age | umur pasien saat terkena diagnosis.
gender | gender pasien (terdiri dari male dan female).
country | negara tempat tinggal pasien.
diagnosis_date | tanggal saat pasien didiagnosis menderita kanker paru.
cancer_stage | stadium kanker paru-paru saat terdiagnosis (terdiri dari Stage I, Stage II, Stage III, Stage IV).
family_history | menunjukkan apakah ada riwayat kanker dalam keluarga (terdiri dari iya (yes), dan tidak (no)).
smoking_status | status merokok (misalnya, perokok aktif (current smoker), mantan perokok (former smoker), tidak pernah merokok (never smoked), dan perokok pasif (passive smoker)).
bmi | indeks massa tubuh pada saat pasien terdiagnosis (indeks bmi normal 18,5 hingga 22,9).
cholesterol_level | kadar kolesterol pasien.
hypertension | menunjukkan pasien menderita tekanan darah tinggi atau hipertensi (terdiri dari ya atau tidak).
asthma | menunjukkan pasien menderita asma (terdiri dari ya atau tidak).
cirrhosis | menunjukkan pasien memiliki sirosis hati (teridiri dari ya atau tidak).
other_cancer | menunjukkan pasien pernah menderita kanker lain selain diagnosis utama, kanker paru (terdiri dari ya atau tidak).
treatment_type | jenis pengobatan yang diterima pasien (teridir dari, pembedahan (surgery), kemoterapi (chemotherapy), radiasi (radiation), dan kombinasi)
end_treatment_date | tanggal terakhir pasien menyelesaikan pengobatan kanker atau meninggal dunia.
survived | menunjukkan apakah pasien selamat (terdiri dari ya dan tidak).

## Exploratory Data Analysis

sebelum memasuki data preparation, perlu dilakukan pemahaman terhadap pola dan persebaran data. Eksplorasi data akan dilakukan secara eksplorasi data deskriptif,  handling outlier dan missing value, univariate data, dan multivariate data.

### Data Descriptive
<img src="https://raw.githubusercontent.com/ayalya/PredictiveAnalysis_LungCancer/main/asset/2DataDescribe.png" align="center"><a></a>

*Gambar 2, Data Deskriptif*

Dari nilai statistik di atas, diketahui bahwa:
1. Tidak ada nilai kosong atau null pada dataset.
2. Pada fitur "Age", terdapat anomali terhadap nilai nimimal penderita kanker berumur 4 tahun sedangkan umur maksimal 104 tahun. Rata-rata penderita kanker berumur 55 tahun.
3. Rata-rata "bmi" pasien kanker paru-paru menyentuh angka 30 yang beratu berat badan pasien obesitas. Nilai normal BMI pada orang dewasa adalah 18,5 hingga 24,9.
4. Rata-rata pasien kanker paru-paru memiliki angka kolesterol yang tinggi mencapai angka 233 yang ditujukan pada fitur "colesterol_level". Kadar kolesterol normal berada di bawah angka 200 ml/Hg.
5. Hampir semua penderita kanker paru-paru memiliki tekanan darah tinggi (hipertensi) dapat dilihat pada fitur "hypertension". Fitur ini merupakan data kategorikal yang telah dilakukan tahapan label encoding dan terbagi menjadi dua nilai: 1 (tinggi) dan 0 (normal). Nilai rata-rata menunjukkan angka 0,75, artinya sekitar 75% pasien dalam dataset memiliki riwayat hipertensi.
6. Struktur data pada fitur "asthma" juga merupakan data kategorikal yang telah dilakukan tahapan label encoding. Nilai rata-rata pada fitur menunjukkan angka 0,47 menunjukkan bahwa hampir setengah penderita kanker paru-paru menderita penyakit asma.
7. Pada fitur "cirrhosis" menujukkan rata-rata 0,226 menunjukkan bahwa hanya sedikit pasien yang menderita kerusakan pada hati.
8. Rata-rata pada fitur "other_cancer" menunjukkan angka 0,088. Artinya hanya sebagian pasien kanker paru-paru yang mengidap kanker lainnya.
9. Pada fitur "survival" menunjukkan rata-rata kesempatan hidup pasien yang selamat sebesar 0,22 atau 22%. Meskipun demikian, masih ada harapan pasien dengan kanker paru-paru untuk selamat dari penyakit ini.

### Mengecek Missing Value, Data Duplicate, and Outlier

Pada proyek ini, dilakukan pengecekan terhadap data yang hilang dan tidak valid. Hasil pemeriksaan menunjukkan bahwa tidak terdapat kolom kosong maupun data duplikat dalam dataset. Selanjutnya, identifikasi outlier dilakukan berdasarkan eksplorasi data deskriptif. Ditemukan adanya outlier pada fitur "age".

<img src="https://raw.githubusercontent.com/ayalya/PredictiveAnalysis_LungCancer/main/asset/3BoxplotBeforeHandlingOutlier.png" align="center"><a></a>

*Gambar 3, Visualisasi Boxplot Sebelum Handling Outlier*

### Univariate Data
#### Data Kategori
Data kategori pada proyek ini meliputi fiur "gender", "country", "cancer_stage", "family_history", "smoking_status", dan "treatment_type". Selanjutnya berikut adalah nilai unik pada masing-masing fitur.

<img src="https://raw.githubusercontent.com/ayalya/PredictiveAnalysis_LungCancer/main/asset/6CatFeature.png" align="center"><a></a>

*Gambar 4, Nilai Unik pada Fitur Kategori*

Untuk mengetahui persebaran diantara fitur, dilakukan visualisasi menggunakan histogram.

<img src="https://raw.githubusercontent.com/ayalya/PredictiveAnalysis_LungCancer/main/asset/7HistogramKategori.png" align="center"><a></a>

*Gambar 5, Histogram pada Fitur Kategori*

Dari Gambar 6, dapat diketahui bahwa:
1. Semua fitur kategorikal memiliki distribusi yang seimbang pada masing-masing nilai uniknya.
2. Pada fitur "gender" pasien laki-laki dan perempuan sama banyaknya.
3. Persebaran negara pada fitur "country" memiliki distribusi yang seimbang atau jumlah pada setiap nilainya sama.
4. Pada fitur "cancer_stage", Sebagian besar pasien kanker paru-paru terdiagnosis pada Stadium IV, diikuti oleh Stadium III, Stadium II, dan paling rendah pada Stadium I.
5. Pasien dengan riwayat anggota keluarga menderita kanker paru-paru memiliki potensi sama besarnya dengan pasien yang keluarganya tidak memiliki riwayat kanker paru-paru.
6. Riwayat kanker paru-paru lebih tinggi pada perokok pasif dan paling rendah pada perokok aktif.
7. Penanganan paling banyak dilakukan pada pasien kanker paru-paru dengan operasi dan kemoterapi, sedangkan yang paling sedikit menggunakan radiasi. Distribusinya pada fitur ini tergolong masih seimbang dan belum bisa menjawab pertanyaan permasalahan pertama dalam hubungan jenis perawatan dengan tingkat keberlangsungan hidup pasien.
8. Pengamatan pertama kali dilakukan pada tahun 2014 yang ditunjukan pada nilai min pada fitur "diagnosis_date".
9. Nilai maksimal "end_treatment_date" terdapat pada tahun 2026, sedangkan proyek saat ini dikerjakan pada tahun 2025. Terdapat kecurigaan salah input pada data. 

#### Data Numerik

Data numerik pada proyek ini adalah "age", "diagnosis_date", "bmi", "colesterol_level", "end_treatment_date", "hypertension", "asthma", "cirrhosis", "other_cancer", dan "survived".

<img src="https://raw.githubusercontent.com/ayalya/PredictiveAnalysis_LungCancer/main/asset/6HistogramKategori.png" align="center"><a></a>

*Gambar 6, Histogram Data Numerik*

Dari grafik di atas, dapat diketahui bahwa:
1. Disrtibusi pada fitur "age" merupakan distribusi normal dan menunjukkan kenaikan nilai di beberapa angka.
2. Distribusi pada fitur "diagnosis_date" menyebar dan mengalami penurunan pasien setelah tahun 2022. Bisa dikatakan persebaran sedikit miring ke kiri atau left-skewd.
3. Persebaran pada fitur "bmi" distribusinya merata diantara 18-45 dan menunjukkan variasi yang luas.
4. Pada fitur "colesterol_level" distribusi memiliki dua puncak atau bimodal setelah nilai sekitar 240. Bisa dikategorikan terdapat kelompok dengan risiko kolesterol tinggi.
5. Pada fitur "hypertension", "asthma", "cirrhosis", "other_cancer" dan "survived" merupakan variabel biner dengan nilai 0 (tidak memiliki kondisi tersebut) dan 1 (memiliki kondisi tersebut). Dapat disimpulkan bahwa:
- "hypertension", lebih banyak pasien dengan kondisi tekanan darah tinggi (nilai 1 paling tinggi).
- "asthma", sebagian besar pasien tidak memiliki penyakit asma.
- "cirrhosis", hanya sebagian kecil pasien memiliki kondisi ini.
- "other_cancer", hanya sebagian kecil pasien memiliki kondisi ini.
- "survived", mayoritas pasien tidak bertahan hidup (nilai 0 paling tinggi).

<img src="https://raw.githubusercontent.com/ayalya/PredictiveAnalysis_LungCancer/main/asset/9DistribusiTarget.png" align="center"><a></a>

*Gambar 7, Pie Chart Persebaran pada Target (survived)*

Target yang akan digunakan pada proyek ini merupakan "survived" yang merupakan data kategorikal yang sudah memasuki tahap label encoding. Memiliki persebaran pasien 22% selamat, sedangkan sisanya tidak selamat.

### Multivariate Data

Muntuk melihat struktur dan karakteristik data, dilakukan visualisasi pairplot.

<img src="https://raw.githubusercontent.com/ayalya/PredictiveAnalysis_LungCancer/main/asset/8Pairplot.png" align="center"><a></a>

*Gambar 8, Visualisasi Pairplot Dataset*

Didapatkan interpretasi, sebagai berikut:
1. Distribusi pada fitur "age" merupakan distribusi normal.
2. Korelasi fitur "colesterol_level" dengan "bmi" menunjukkan miring ke kanan atau right-skewd menunjukkan bahwa korelasi yang positif terjadi di antara kedua fitur tersebut.
3. Selain dari dari ketiga fitur di atas, merupakan fitur kategorikal yang diubah menjadi numerik sehingga korelasinya merupakan garis lurus.
4. Dapat dipastikan, karakteristik dataset merupakan non-linear dan tersebar.

## Data Preparation
Sebelum model dibuat, berikut langkah-langkah yang dilakukan:

### 1. Memastikan Tipe Data Datetime

Pada model identifikasi tipe data sebelumnya, fitur "diagnosis_date" dan "end_treatment_date" merupakan tipe data object. Pada tahap ini, kedua fitur tersebut diubah tipe datanya menjadi datatime dengan format Tahun-Bulan-Hari. Proses ini dilakukan untuk memfilter data dan penambahan fitur pada tahapan Data Preparation selanjutnya.

<img src="https://raw.githubusercontent.com/ayalya/PredictiveAnalysis_LungCancer/main/asset/9DataDescribeAfterDatetime.png" align="center"><a>

*Gambar 9, Data Deskriptif Setelah Mengubah Datetime*

### 2. Filtered Data

Pada info statistik data sebelumnya, terdapat nilai maksimal pada fitur "end_treatment_date" di tahun 2026. Padahal data dipublish pada akhir tahun kemarin, maka dari itu baik pada fitur "diagnosis_date" dan "end_treatment_date" akan mengambil data sebelum tanggal 31 Desember 2024.

### 3. Menangani Outlier
Proses visualisasi data univariate data numerik, fitur "age" memiliki banyak outlier. Untuk itu, diperlukan mengatasi outlier dengan menggunakan metode Interquantile Range (IQR). Cara kerja IQR dengan menghitung selisih antara kuartil atas (Q3) dan kuartil bawah (Q1).

<img src="https://raw.githubusercontent.com/ayalya/PredictiveAnalysis_LungCancer/main/asset/4BoxplotAfterHandlingOutlier.png" align="center"><a></a>

*Gambar 4, Visualisasi Boxplot Sesudah Handling Outlier*

<img src="https://raw.githubusercontent.com/ayalya/PredictiveAnalysis_LungCancer/main/asset/5DataDescribeAfterHandlingOutlier.png" align="center"><a>

*Gambar 10, Data Deskriptif Setelah Handling Outlier*

Setelah menangani outlier dan filter data datetime, didapatkan informasi:
1. Rata-rata umur penderita berada di rentang 48-62. Dapat disimpulkan tidak ada outlier.
2. Indeks masa tubuh atau "bmi" pasien rata-rata berada di rentang 23-38.
3. Rata-rata rentang nilai kolesterol pasien berada di rentang 196-271.
4. Akhir masa perawatan berada pada tanggal 30 Desember 2024.
1. Jumlah data setelah filter data dan menangani outlier adalah 826.352 baris data. 

### 4. Feature Engineering

Menambahkan beberapa fitur yang mungkin relevan agar model dapat memberikan klasifikasi dengan performa yang baik. Hal ini berfokus untuk menangani data yang terlalu general dan mengharapkan dapat meningkatkan akurasi pada model. Berikut fitur yang akan ditambahkan:
1. "survival_months", merupakan selisih dari "diagnosis_date" dengan "end_treatment_date" yang disajikan dalam bentuk bulan.
2. "survival_group", hasil binning dari "survival_months" yang disajikan dalam bentuk interval 2 tahun.
3. "age_group", hasil binning dari "age" dengan membagi menjadi rentang puluhan, misalnya usia 20an, 30an, dan seterusnya.
4. "cholesterol_group", hasil binning dari "cholesterol_level" yang dibagi menjadi Normal jika di bawah angka 200 dan High jika kadar kolesterol di atas 200.
5. "bmi_group", hasil binning dari "bmi" yang dibagi menjadi Underweight (<18), Normal (18,5-24,9), dan Overwight (>25).

<img src="https://raw.githubusercontent.com/ayalya/PredictiveAnalysis_LungCancer/main/asset/11CategoricalAfterFineTuning.png" align="center"><a></a>

*Gambar 11, Visualisasi Survival Month Terhadap Cancer Stage*

Setelah melakukan feature engineering, hampir semua fitur kategori terhadap "survival_months" memiliki distribusi yang sama, kecuali pada fitur "cancer_stage". Semakin rendah stadium pada pasien, maka perawatannya lebih lama. Hal ini memberikan interpretasi bahwa kanker dengan stadium semakin tinggi, semakin kecil harapan untuk selamat sesuai dengan distribusi pada fitur "survival". Lamanya pasien bertahan hidup, tidak terlalu berpengaruh terhadap jenis penanganan dan perawatan yang didapatkan karena distribusinya memiliki distribusi yang sama.

### 5. Split Dataset

Pembagian dataset dilakukan menjadi 70% dataset pelatihan atau training dan 30% dataset merupakan data uji atau training. Pemisahan dataset ini dilakukan untuk memisahkan data untuk tahap pelatihan dan uji agar data tidak tercampur dan model berjalan optimal. Tahapan pemisahan ini mengecualikan fitur "diagnosis_date" dan "end_treatment_date" karena sudah diwakilkan oleh "survival_months" dan "survival_group". Sedangkan target yang digunakan pada model klasifikasi adalah "survived".

### 6. Label Encoding

Mengubah nilai data kategorikal menjadi representasi numerik dilakukan agar komputer dapat lebih mudah mengenali fitur tersebut. Proses label encoding dilakukan secara manual pada fitur supaya lebih mudah dipantau. Proses ini dilakukan setelah proses spliting dataset (splitting) dilakukan untuk mencegah terjadinya kebocoran data (data leakage) pada model. Fitur-fitur yang melalui tahapan label encoding, yaitu "gender", "cancer_stage","survival_group", "country", "family_history", "smoking_status", "treatment_type", "age_group", "cholesterol_group", "bmi_group".

#### 7. Normalisasi Data
Normalisasi data dilakukan mennggunakan standard scaler pada fitur numerik terutama bagi data yang memiliki selisih yang tinggi. Tujuannya untuk memastikan data memiliki nilai kecil tanpa mengubah variance dan membantu komputasi lebih ringan. Nilai numerik yang dilakukan normalisasi, yaitu "age", "bmi", "cholesterol_level", dan "survival_months".

#### 8. Feature Selection

Tidak semua fitur akan digunakan pada proses pembangunan model, mengingat terdapat 18 fitur pada data. Tujuannya agar model bisa lebih mengenali pola yang penting pada data berdasarkan korelasi dan kontribusinya terhadap model. Maka akan dilakukan seleksi fitur menggunakan teknik Lasso dan mempertimbangkan matriks korelasi (correlation matrix). 

<img src="https://raw.githubusercontent.com/ayalya/PredictiveAnalysis_LungCancer/main/asset/12CorrelationMatrix.png" align="center"><a></a>

*Gambar 12, Heatmap Hubungan Antar Fitur Setelah Data Mengalami Preprocessing*

Dari korelasi matriks di atas, terdapat beberapa fitur yang saling berkorelasi dengan nilai yang cukup tinggi, yaitu:
1. Fitur "age" dengan "age_group" memiliki korelasi negatif.
2. "bmi", "bmi_group", "cholesterol_level", dan "colesterol_group" memiliki korelasi dengan nilai yang cukup signifikan dan dapat mempengaruhi model.
3. "survival_month" dan "survival_group" memiliki korelasi yang positif.
4. Maka dari itu, ke delapan fitur tersebut dapat dijadikan bahan pertimbangan untuk dimasukkan ke dalam model.

Banyaknya jumlah fitur, distribusi yang hampir merata pada sebagian besar fitur, serta korelasi antar fitur yang rendah dapat menyebabkan model membuat prediksi yang terlalu umum (general). Oleh karena itu, dilakukan proses pemilihan fitur (feature selection) untuk meningkatkan kinerja model menggunakan teknik Lasso.

**Teknik Lasso** (Least Absolute Shrinkage and Selection Operator) merupakan salah satu teknik regresi dalam statistika dan machine learning untuk pilihan fitur. Lasso menambahkan penalti berupa nilai absolut ke dalam fungsi loss, yang disebut dengan L1 regularization. Bekerja dengan membuat beberapa koefisien regresi menjadi nol sehingga fitur yang tidak penting dihilangkan dari model.

Kelebihan dari Lasso:
1. Dapat melakukan seleksi fitur pada model.
2. Membantu menangani overfitting.
3. Cocok untuk dataset dengan banyak fitur, terutama jika beberapa fitur tidak penting digunakan.

Pada teknik Lasso yang memiliki hubungan dengan target, treshold yang ditetapkan 0,001 dan mendapatkan hasil fitur "counry", "cancer_stage", dan "age_group". Ditambah dari korelasi yang tinggi antar pada fitur "survival_months", "bmi", dan "cholesterol_group" dari visualisasi matrix correlation. Diputuskan, total 6 fitur yang akan digunakan dalam proses pembangunan model ini. Fitur-fitur yang dipilih, yaitu "country", "cancer_stage", "age_group", "survival_months", "bmi", dan "cholesterol_group".

## Modeling

Berdasarkan karakteristik data yang non-linear serta jumlah data yang besar, dibutuhkan model dan algoritma yang cukup kompleks dan mampu menangani data dalam jumlah besar. Oleh karena itu, dilakukan percobaan menggunakan empat model berikut:

### **1. Decision Tree**

Model Decision Tree bekerja dengan membentuk struktur pohon keputusan, di mana setiap cabang mewakili keputusan berdasarkan fitur tertentu, dan setiap daun mewakili output/klasifikasi. Dalam proyek ini, model menggunakan pemisahan berdasarkan Entropy untuk memilih fitur terbaik di setiap node. Karakteristik data yang terdiri dari fitur numerik dan kategorikal menjadikan model ini cukup cocok digunakan. Namun, Decision Tree cenderung overfitting dan kurang stabil terhadap perubahan data, terutama pada dataset besar. Parameter yang digunakan dalam proyek ini "random_state = 42" dan "max_depth = 10". 

### **2. Random Forest**

Random Forest merupakan pengembangan dari Decision Tree yang menggunakan teknik bagging, yaitu membangun banyak pohon keputusan secara acak dari subset data dan fitur yang berbeda. Setiap pohon memberikan prediksi, dan hasil akhir ditentukan dengan voting mayoritas. Hal ini membuat model lebih stabil, akurat, dan tahan terhadap overfitting dibandingkan Decision Tree tunggal. Parameter yang digunakan pada proyek ini adalah "random_forest = 42" dan "n_estimators = 50". Hasilnya, model menjadi lebih stabil dan akurat dibandingkan Decision Tree meskipun waktu latih menjadi lebih lama.


### **3. K-Nearest Neighbor**

KNN adalah model berbasis instance-based learning yang memprediksi kelas berdasarkan sejumlah tetangga terdekat (dalam ruang fitur). Model ini menggunakan pengukuran jarak, dalam hal ini Minkowski Distance, untuk menentukan kedekatan antar data. Cocok digunakan untuk data non-linear seperti pada proyek ini, namun sensitif terhadap skala fitur sehingga memerlukan normalisasi. Model ini juga memiliki kelemahan dalam efisiensi karena proses prediksi memerlukan pencarian seluruh tetangga. Parameter yang digunakan "n_neighbors=5".

### **4. Gradient Boosing**

Gradient Boosting adalah algoritma ensemble boosting yang membangun model secara bertahap, dengan setiap model baru mempelajari dan memperbaiki kesalahan dari model sebelumnya. Berbeda dari Random Forest yang membangun pohon secara paralel, Gradient Boosting membangunnya secara berurutan (iteratif), fokus pada mengurangi error (residual) dari prediksi sebelumnya. Model ini sangat efektif untuk data kompleks dan non-linear serta memiliki performa tinggi jika dituning dengan baik. Dalam proyek ini, parameter yang digunakan adalah "n_estimators = 50" dan "random_state=42". Data yang besar, non-liear, dan kompleks merupakan karakteristik data ini yang dapat diatasi oleh Gradient Boosting meskipun membutuhkan waktu dengan kompleksitas yang tinggi.

Proses data preparation sangat membantu proses modeling dan setiap langkahnya sangat berkaitan. Penambahan beberapa fitur membantu dalam menambahkan korelasi pada data, label encoding dan normalisasi data membantu data dikenali oleh model dan bekerja dengan lebih cepat. Pelaksanannya dilakukan setelah spliting data membantu model tidak bocor antara training dan testing. Pemilihan fitur melalui teknik Lasso dan mempertimbangkan *confussion matrix* membantu dalam reduksi data, mengurangi kompleksitas model, dan meningkatkan akurasi dengan menghindari data dengan korelasi yang tinggi. Penggunaan teknik ini juga secara langsung menjawab pernyataan permasalahan atas fitur apa saja yang berpengaruh kepada keselamatan dan diagnosis kanker paru-paru.


## Evaluation

Metrik yang digunakan pada proyek ini meliputi:
1. **Accuracy**  
   Persentase prediksi yang benar dari keseluruhan data:

   ![Accuracy](https://latex.codecogs.com/png.image?\dpi{120}&space;\text{Accuracy}=\frac{TP+TN}{TP+TN+FP+FN})

2. **Precision**  
   Persentase seberapa banyak prediksi sebagai positif yang benar-benar positif:

   ![Precision](https://latex.codecogs.com/png.image?\dpi{120}&space;\text{Precision}=\frac{TP}{TP+FP})

3. **Recall**  
   Persentase data yang benar-benar positif yang berhasil terdeteksi oleh model:

   ![Recall](https://latex.codecogs.com/png.image?\dpi{120}&space;\text{Recall}=\frac{TP}{TP%20+%20FN})

4. **F1-Score**  
   Rata-rata harmonis dari Precision dan Recall. Berguna saat data tidak seimbang:

   ![F1 Score](https://latex.codecogs.com/png.image?\dpi{120}&space;F1\_Score=2\cdot\frac{Precision\cdot%20Recall}{Precision+Recall})

Berikut adalah evaluasi pada model:


*Tabel 2, Matriks Evaluasi Model Training*
Model | Accuracy | Precission | Recall | F1 Score
----------|----------|----------|----------|----------
Decision Tree | 0,779903 | 0,716814 | 0,001483 | 0,002960
Random Forest | 0,811924 | 0,674910 | 0,282171 | 0,397960
KNN | 0,789898 | 0,568168 | 0,192832 | 0,287939
Gradient Boosting | 0,7779705 | 0,000000 | 0,000000 | 0,000000

*Tabel 3, Matriks Evaluasi Model Testing*
Model | Accuracy | Precission | Recall | F1 Score
----------|----------|----------|----------|----------
Decision Tree | 0,779624 | 0,236842 | 0,000495 | 0,0000987
Random Forest | 0,719856 | 0,211921 | 0,100269 | 0,136130
KNN     | 0,736785 | 0,209055 | 0,070309 | 0,105229
Gradient Boosting | 0,779866 | 0,000000 | 0,000000 | 0,000000

Proses evaluasi pada proyek ini akan menggunakan akurasi, confusion matrix, dan f1-score. Mengingat rasio nilai pada target tidak merata dengan nilai 1 lebih sedikit, maka evaluasi model akan lebih memperhatikan prediksi ke nilai 1 atau true positif (maka dari itu metrik f1-score lebih cocok).

**1. Decision Tree**, Akurasi pada pelatihan dan prediksi data 0,779. Dari gambar di bawah, hanya 9 data yang diprediksi sebagai true positive. Diperoleh skor F1 pada prediksi 0,000987. 
<br>
<img src="https://raw.githubusercontent.com/ayalya/PredictiveAnalysis_LungCancer/main/asset/12CMDT.png" align="center"><a>

*Gambar13, Confussion Matrix Decision Tree*
</br>

**2. Random Forest**, terdapat 1.824 data yang dapat diprediksi sebagai true positif pada model ini. Model ini mendapatkan akurasi pelatihan 0,81 dan akurasi prediksi 0,719 sedangkan nilai F1 pada prediksi 0,136.
<br>
<img src="https://raw.githubusercontent.com/ayalya/PredictiveAnalysis_LungCancer/main/asset/13CMRF.png" align="center"><a>

*Gambar14, Confussion Matrix Random Forest*
</br>

**3. KNN**, dapat memprediksi 1.279 data sebagai true positif. Mendapatkan akurasi pelatihan 0,789 dan akurasi prediksi 0,736. Model mendapatkan skor f1 pada pelatihan 0,287 dan prediksi 0,1052.
<br>
<img src="https://raw.githubusercontent.com/ayalya/PredictiveAnalysis_LungCancer/main/asset/13CMKNN.png" align="center"><a>

*Gambar15, Confussion Matrix KNN*
</br>

**4. Gradient Boosting**, model ini tidak dapat mengenali data true positif sehingga mendapatkan skor f1 pada pelatihan dan pengujian adalah 0. Namun masih bisa memprediksi nilai true false, memiliki akurasi pelatihan dan pengujian sebesar 0,779.
<br>
<img src="https://raw.githubusercontent.com/ayalya/PredictiveAnalysis_LungCancer/main/asset/14CMGB.png" align="center"><a>

*Gambar16, Confussion Matrix Gradient Boosting*
</br>

Akurasi dan nilai F1 pada keseluruhan model belum menunjukkan hasil yang cukup baik menyentuh angka rata-rata belum menyentuh angka 0,8 dan nilai F1 tidak sampai 0,5. Hal ini dapat terjadi karena:
1. Persebaran pada data kategorikal terlalu menyebar dan model terlalu menggeneralisasi sehingga sulit untuk mengenali data, bahkan overfitting.
2. Meskipun persebaran data merata, target pada proyek ini yaitu "survived" memiliki sebaran yang tidak seimbang atau undersampling. Kasus pasien yang tidak selamat lebih banyak dibandingkan dengan pasien yang hidup, sehingga model lebih mengenali nilai True Negative atau pasien yang tidak selamat.

### Hasil Evaluasi
Untuk membandingkan seluruh nilai akurasi, berikut adalah perbandingan nilai akurasi pada model:
<img src="https://raw.githubusercontent.com/ayalya/PredictiveAnalysis_LungCancer/main/asset/15accuracy.png" align="center"><a>

*Gambar17, Confussion Matrix Random Forest*

Berdasarkan gambar perbandingan akurasi di atas, model dengan akurasi tertinggi adalah Random Forest dan KNN. Selain akurasi, nilai F1 pada pelatihan dan prediksi merupakan nilai teringgi dengan jumlah kesalahan prediksi paling minimum dan paling banyak memprediksi nilai true positif, terutama pada model Random Forest.

## Conclusion
Berdasarkan hasil yang diperoleh pada proses EDA dan pengujian model terbaik untuk menentukan kejadian kanker paru-paru dapat menjawab pertanytaan permasalahan dan dapat disimpulkan bahwa:
1. Beberapa fitur yang terbukti paling berpengaruh terhadap diagnosis dan keselamatan pasien kanker paru adalah: "country", "cancer_stage", "age_group", "survival_months", "bmi", dan "cholesterol_group". Fitur country mencerminkan kondisi fasilitas kesehatan yang tersedia, sedangkan stadium kanker ("cancer_stage") menunjukkan korelasi kuat dengan harapan hidup—semakin tinggi stadiumnya, semakin rendah kemungkinan keselamatan pasien. Hal ini memperkuat pentingnya deteksi dini dan pola hidup sehat.
2. Ditemukan adanya korelasi antara keberlangsungan hidup dengan stadium kanker. Pasien dengan stadium yang lebih rendah cenderung memiliki survival rate yang lebih tinggi. Namun, jenis perawatan (treatment type) tidak menunjukkan hubungan signifikan dengan stadium, menandakan bahwa efektivitas pengobatan bisa sangat bervariasi dan perlu disesuaikan secara individual. Literatur menunjukkan bahwa operasi adalah pilihan utama, tetapi harus dipertimbangkan berdasarkan kondisi spesifik pasien.
3. Dari pengujian terhadap empat model machine learning: Decision Tree, Random Forest, KNN, dan Gradient Boosting, diperoleh dua model paling optimal:
- **Random Forest** menunjukkan performa terbaik dari segi akurasi dan kestabilan prediksi secara keseluruhan.
- **KNN** memberikan nilai F1 Score tertinggi untuk kelas positif, menunjukkan kemampuannya dalam mendeteksi pasien yang memiliki kemungkinan tinggi untuk bertahan hidup.

## Referensi
<a name="ref1"></a>[1] J. Joseph and L. W. A. Rotty, “Kanker Paru: Laporan Kasus”, MSJ, vol. 2, no. 1, Jul. 2020.

<a name="ref2"></a>[2] Purangga, Tengku Arya, "Deteksi Kanker Paru-Paru Menggunakan Machine Learning dengan Metode Artificial Neural Network dan Decision Tree" (Skripsi), Fakultas Sains dan Teknologi. UIN Syarif Hidayatullah Jakarta, 2024.

<a name="ref3"></a>[3] Mitra Keluarga, "Penyebab Utama Kanker Paru dan Pengobatan yang Tepat. Terapi di Oncology Center!", Mitra Keluarga. Online. Tersedia: https://www.mitrakeluarga.com/artikel/kanker-paru. [Diakses: 17 Mei 2025].

<a name="ref4"></a>[4] Wulandari, Laksmi, "Terapi Target pada Kanker Paru". Surabaya: Universitas Airlangga Press, 2019.

<a name="ref5"></a>[5] "Hari Kanker Sedunia 2025", RS Roemani Muhammadiah. Online. Tersedia: https://rsroemani.com/rv2/hari-kanker-sedunia-2025-united-by-unique. [Diakses: 17 Mei 2025].

[Github](https://github.com/ayalya/PredictiveAnalysis_LungCancer/tree/main)