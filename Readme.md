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

<img src="https://github.com/ayalya/PredictiveAnalysis_LungCancer/blob/main/asset/1Datatype.png" align="center"><br>

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
<img src="https://github.com/ayalya/PredictiveAnalysis_LungCancer/blob/main/asset/2DataDescribe.png" align="center"><br>

*Gambar 2, Data Deskriptif*

Pada informasi statistik pada tipe data numerikal dan tipe datetime di atas, didapatkan informasi bahwa:
1. Tidak ada nilai kosong atau null pada dataset.
2. Pada fitur `Age`, terdapat anomali terhadap nilai nimimal penderita kanker berumur 4 tahun sedangkan umur maksimal 104 tahun. Rata-rata penderita kanker berumur 55 tahun.
3. Pengamatan pertama kali dilakukan pada tahun 2014 yang ditunjukan pada nilai min pada fitur `diagnosis_date`.
4. Nilai maksimal `end_treatment_date` terdapat pada tahun 2026, sedangkan proyek saat ini dikerjakan pada tahun 2025. Terdapat kecurigaan salah input pada data.
5. Rata-rata `bmi` pasien kanker paru-paru menyentuh angka 30 yang beratu berat badan pasien obesitas. Nilai normal BMI pada orang dewasa adalah 18,5 hingga 24,9.
6. Rata-rata pasien kanker paru-paru memiliki angka kolesterol yang tinggi mencapai angka 233 yang ditujukan pada fitur `colesterol_level`. Kadar kolesterol normal berada di bawah angka 200 ml/Hg.
7. Hampir semua penderita kanker paru-paru memiliki tekanan darah tinggi (hipertensi) dapat dilihat pada fitur `hypertension`. Fitur ini merupakan data kategorikal yang telah dilakukan tahapan label encoding dan terbagi menjadi dua nilai: 1 (tinggi) dan 0 (normal). Nilai rata-rata menunjukkan angka 0,75, artinya sekitar 75% pasien dalam dataset memiliki riwayat hipertensi.
8. Struktur data pada fitur `asthma` juga merupakan data kategorikal yang telah dilakukan tahapan label encoding. Nilai rata-rata pada fitur menunjukkan angka 0,47 menunjukkan bahwa hampir setengah penderita kanker paru-paru menderita penyakit asma.
9. Pada fitur `cirrhosis` menujukkan rata-rata 0,226 menunjukkan bahwa hanya sedikit pasien yang menderita kerusakan pada hati.
10. Rata-rata pada fitur `other_cancer` menunjukkan angka 0,088. Artinya hanya sebagian pasien kanker paru-paru yang mengidap kanker lainnya.
11. Pada fitur `survival` menunjukkan rata-rata kesempatan hidup pasien yang selamat sebesar 0,22 atau 22%. Meskipun demikian, masih ada harapan pasien dengan kanker paru-paru untuk selamat dari penyakit ini.

### Handling Missing Value, Data Duplicate, and Outlier

Pada proyek ini, dilakukan pengecekan terhadap data yang hilang dan tidak valid. Hasil pemeriksaan menunjukkan bahwa tidak terdapat kolom kosong maupun data duplikat dalam dataset. Selanjutnya, identifikasi outlier dilakukan berdasarkan eksplorasi data deskriptif. Ditemukan adanya outlier pada dua fitur, yaitu `age` dan `end_treatment_date`.
- Pada fitur `end_treatment_date`, terdapat melebihi tanggal publish dataset. Oleh harena itu, akan dilakukan pembatasan nilai maksimum hingga tanggal 31 Desember 2024.
- Sementara itu, pada fitur `age` terindikasi adanya outlier yang ditandai oleh rentang minimum dan maksimum yang sangat lebar. Penanganan outlier pada fitur ini digunakan metode Interquantile Range (IQR).

<img src="https://github.com/ayalya/PredictiveAnalysis_LungCancer/blob/main/asset/3BoxplotBeforeHandlingOutlier.png" align="center"><a>

*Gambar 3, Visualisasi Boxplot Sebelum Handling Outlier*

<img src=https://github.com/ayalya/PredictiveAnalysis_LungCancer/blob/main/asset/4BoxplotBeforeAfterOutlier.png align="center"><a>

*Gambar 4, Visualisasi Boxplot Sesudah Handling Outlier*

<img src="https://github.com/ayalya/PredictiveAnalysis_LungCancer/blob/main/asset/5DataDescribeAfterHandlingOutlier.png" align="center"><a>

*Gambar 5, Data Deskriptif Setelah Handling Outlier*

Setelah proses penanganan outlier pada kedua fitur tersebut, jumlah data tersisa sebanyak 826.352 baris. Hasilnya menunjukkan bahwa selisih nilai pada fitur `age` menjadi lebih wajar, dan tidak terdapat lagi nilai `end_treatment_date` yang melebihi tanggal publikasi dataset.

### Univariate Data
#### Data Kategori
Data kategori pada proyek ini meliputi fiur `gender`, `country`, `cancer_stage`, `family_history`, `smoking_status`, dan `treatment_type`. Selanjutnya berikut adalah nilai unik pada masing-masing fitur.

<img src="https://github.com/ayalya/PredictiveAnalysis_LungCancer/blob/main/asset/6CatFeature.png" align="center"><a>

*Gambar 6, Nilai Unik pada Fitur Kategori*

Untuk mengetahui persebaran diantara fitur, dilakukan visualisasi menggunakan histogram.

<img src="https://github.com/ayalya/PredictiveAnalysis_LungCancer/blob/main/asset/7HistogramKategori.png" align="center"><a>

*Gambar 7, Histogram pada Fitur Kategori*

Dari Gambar 6, dapat diketahui bahwa:
1. Semua fitur kategorikal memiliki distribusi yang seimbang pada masing-masing nilai uniknya.
2. Pada fitur `gender` pasien laki-laki dan perempuan sama banyaknya.
3. Persebaran negara pada fitur `country` memiliki distribusi yang seimbang atau jumlah pada setiap nilainya sama.
4. Pada fitur `cancer_stage`, Sebagian besar pasien kanker paru-paru terdiagnosis pada Stadium IV, diikuti oleh Stadium III, Stadium II, dan paling rendah pada Stadium I.
5. Pasien dengan riwayat anggota keluarga menderita kanker paru-paru memiliki potensi sama besarnya dengan pasien yang keluarganya tidak memiliki riwayat kanker paru-paru.
6. Riwayat kanker paru-paru lebih tinggi pada perokok pasif dan paling rendah pada perokok aktif.
7. Penanganan paling banyak dilakukan pada pasien kanker paru-paru dengan operasi dan kemoterapi, sedangkan yang paling sedikit menggunakan radiasi. Distribusinya pada fitur ini tergolong masih seimbang dan belum bisa menjawab pertanyaan permasalahan pertama dalam hubungan jenis perawatan dengan tingkat keberlangsungan hidup pasien.

#### Data Numerik

Data numerik pada proyek ini adalah `age`, `diagnosis_date`, `bmi`, `colesterol_level`, `end_treatment_date`, `hypertension`, `asthma`, `cirrhosis`, `other_cancer`, dan `survived`.

<img src="https://github.com/ayalya/PredictiveAnalysis_LungCancer/blob/main/asset/8HistogramKategori.png" align="center"><a>

*Gambar 8, Histogram Data Numerik*

Dari grafik di atas, dapat diketahui bahwa:
1. Disrtibusi pada fitur `age` merupakan distribusi normal dan menunjukkan kenaikan nilai di beberapa angka.
2. Distribusi pada fitur `diagnosis_date` menyebar dan mengalami penurunan pasien setelah tahun 2022. Bisa dikatakan persebaran sedikit miring ke kiri atau left-skewd.
3. Persebaran pada fitur `bmi` distribusinya merata diantara 18-45 dan menunjukkan variasi yang luas.
4. Pada fitur `colesterol_level` distribusi memiliki dua puncak atau bimodal setelah nilai sekitar 240. Bisa dikategorikan terdapat kelompok dengan risiko kolesterol tinggi.
5. Pada fitur `end_treatment_date` distribusinya mirip `diagnosis_date`, pasien mengalami kenaikan nilai setelah tahun 2015.
6. Pada fitur `hypertension`, `asthma`, `cirrhosis`, `other_cancer` dan `survived` merupakan variabel biner dengan nilai 0 (tidak memiliki kondisi tersebut) dan 1 (memiliki kondisi tersebut). Dapat disimpulkan bahwa:
- `hypertension`, lebih banyak pasien dengan kondisi tekanan darah tinggi (nilai 1 paling tinggi).
- `asthma`, sebagian besar pasien tidak memiliki penyakit asma.
- `cirrhosis`, hanya sebagian kecil pasien memiliki kondisi ini.
- `other_cancer`, hanya sebagian kecil pasien memiliki kondisi ini.
- `survived`, mayoritas pasien tidak bertahan hidup (nilai 0 paling tinggi).

<img src="https://github.com/ayalya/PredictiveAnalysis_LungCancer/blob/main/asset/9DistribusiTarget.png" align="center"><a>

*Gambar 9, Pie Chart Persebaran pada Target (survived)*

Target yang akan digunakan pada proyek ini merupakan `survived` yang merupakan data kategorikal yang sudah memasuki tahap label encoding. Memiliki persebaran pasien 22% selamat, sedangkan sisanya tidak selamat.

### Multivariate Data

Muntuk melihat struktur dan karakteristik data, dilakukan visualisasi pairplot.

<img src="https://github.com/ayalya/PredictiveAnalysis_LungCancer/blob/main/asset/10Pairplot.png" align="center"><a>

*Gambar 10, Visualisasi Pairplot Dataset*

Didapatkan interpretasi, sebagai berikut:
1. Distribusi pada fitur `age` merupakan distribusi normal.
2. Korelasi fitur `colesterol_level` dengan `bmi` menunjukkan miring ke kanan atau right-skewd menunjukkan bahwa korelasi yang positif terjadi di antara kedua fitur tersebut.
3. Selain dari dari ketiga fitur di atas, merupakan fitur kategorikal yang diubah menjadi numerik sehingga korelasinya merupakan garis lurus.
4. Dapat dipastikan, karakteristik dataset merupakan non-linear dan tersebar.

## Data Preparation
Sebelum model dibuat, berikut langkah-langkah yang dilakukan:

### 1. Feature Engineering

Menambahkan beberapa fitur yang mungkin relevan agar model dapat memberikan klasifikasi dengan performa yang baik. Hal ini berfokus untuk menangani data yang terlalu general dan mengharapkan dapat meningkatkan akurasi pada model. Berikut fitur yang akan ditambahkan:
1. `survival_months`, merupakan selisih dari `diagnosis_date` dengan `end_treatment_date` yang disajikan dalam bentuk bulan.
2. `survival_group`, hasil binning dari `survival_months` yang disajikan dalam bentuk interval 2 tahun.
3. `age_group`, hasil binning dari `age` dengan membagi menjadi rentang puluhan, misalnya usia 20an, 30an, dan seterusnya.
4. `cholesterol_group`, hasil binning dari `cholesterol_level` yang dibagi menjadi Normal jika di bawah angka 200 dan High jika kadar kolesterol di atas 200.
5. `bmi_group`, hasil binning dari `bmi` yang dibagi menjadi Underweight (<18), Normal (18,5-24,9), dan Overwight (>25).

<img src="https://github.com/ayalya/PredictiveAnalysis_LungCancer/blob/main/asset/11CategoricalAfterFineTuning.png" align="center"><a>

*Gambar 11, Visualisasi Survival Month Terhadap Cancer Stage*

Setelah melakukan feature engineering, hampir semua fitur kategori terhadap `survival_months` memiliki distribusi yang sama, kecuali pada fitur `cancer_stage`. Semakin rendah stadium pada pasien, maka perawatannya lebih lama. Hal ini memberikan interpretasi bahwa kanker dengan stadium semakin tinggi, semakin kecil harapan untuk selamat sesuai dengan distribusi pada fitur `survival`. Lamanya pasien bertahan hidup, tidak terlalu berpengaruh terhadap jenis penanganan dan perawatan yang didapatkan karena distribusinya memiliki distribusi yang sama.

### 2. Split Dataset

Pembagian dataset dilakukan menjadi 70% dataset pelatihan atau training dan 30% dataset merupakan data uji atau training. Pemisahan dataset ini dilakukan untuk memisahkan data untuk tahap pelatihan dan uji agar data tidak tercampur dan model berjalan optimal. Tahapan pemisahan ini mengecualikan fitur `diagnosis_date` dan `end_treatment_date` karena sudah diwakilkan oleh `survival_months` dan `survival_group`. Sedangkan target yang digunakan pada model klasifikasi adalah `survived`.

### 3. Label Encoding

Mengubah nilai data kategorikal menjadi representasi numerik dilakukan agar komputer dapat lebih mudah mengenali fitur tersebut. Proses label encoding dilakukan secara manual pada fitur supaya lebih mudah dipantau. Proses ini dilakukan setelah proses spliting dataset (splitting) dilakukan untuk mencegah terjadinya kebocoran data (data leakage) pada model. Fitur-fitur yang melalui tahapan label encoding, yaitu `gender`, `cancer_stage`,`survival_group`, `country`, `family_history`, `smoking_status`, `treatment_type`, `age_group`, `cholesterol_group`, `bmi_group`.

#### 4. Normalisasi Data
Normalisasi data dilakukan mennggunakan standard scaler pada fitur numerik terutama bagi data yang memiliki selisih yang tinggi. Tujuannya untuk memastikan data memiliki nilai kecil tanpa mengubah variance dan membantu komputasi lebih ringan. Nilai numerik yang dilakukan normalisasi, yaitu `age`, `bmi`, `cholesterol_level`, dan `survival_months`.

#### 5. Feature Selection

Tidak semua fitur akan digunakan pada proses pembangunan model, mengingat terdapat 18 fitur pada data. Tujuannya agar model bisa lebih mengenali pola yang penting pada data berdasarkan korelasi dan kontribusinya terhadap model. Maka akan dilakukan seleksi fitur menggunakan teknik Lasso dan mempertimbangkan matriks korelasi (correlation matrix). 

<img src="https://github.com/ayalya/PredictiveAnalysis_LungCancer/blob/main/asset/12CorrelationMatrix.png" align="center"><a>

*Gambar 12, Heatmap Hubungan Antar Fitur Setelah Data Mengalami Preprocessing*

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

Pada teknik Lasso yang memiliki hubungan dengan target, treshold yang ditetapkan 0,001 dan mendapatkan hasil fitur `counry`, `cancer_stage`, dan `age_group`. Ditambah dari korelasi yang tinggi antar pada fitur `survival_months`, `bmi`, dan `cholesterol_group` dari visualisasi matrix correlation. Diputuskan, total 6 fitur yang akan digunakan dalam proses pembangunan model ini. Fitur-fitur yang dipilih, yaitu `country`, `cancer_stage`, `age_group`, `survival_months`, `bmi`, dan `cholesterol_group`.

## 6. Modeling

Berdasarkan karakteristik data yang tidak linear serta jumlah data yang besar, dibutuhkan model dan algoritma yang cukup kompleks dan mampu menangani data dalam jumlah besar. Oleh karena itu, dilakukan percobaan menggunakan empat model berikut:

### 1. **Decision Tree**
Merupakan model yang sederhana namun cukup efektif pada data yang banyak baik pada fitur kategorikal, maupun numerikal. Interpretasi yang diberikan jelas terhadap proses pengambilan keputusan. Memiliki kelebihan mudah dipahami, dijelaskan melalui visualisasi pohon keputusan, dan dapat menangani data numerik dan kategorikal. Namun, sensitif terhadap perubahan kecil dalam data dan kurang stabil pada data yang kompleks. Hal ini menjadikan algoritma cocok karena memiliki banyak data numerik dan kategorikal

Pemodelan dilakukan menggunakan library `sklearn.ensemble` dengan memasukkan `X_train_selected` dan `y_train` untuk melatih model, lalu menggunakan `X_test_scaled` dan `y_test` untuk menguji model dengan data testing yang tidak ada pada data training. Pada awal pembangunan mode, parameter yang digunakan `random_state` dan `max_depth` hal ini membuat model kurang memberikan hasil yang baik. Parameter yang digunakan setelah melakukan tuning parameter, yaitu `max_depth = 8` merupakan kedalaman maksimum pohon, `min_samples_leaf = 1` merupakan jumlah minimum sampel di daun, dan `splitter = best` strategi pemilihan split di setiap node.

### 2. **Random Forest**
Merupakan pengembangan dari model Decision Tree dengan teknik ensemble learning dalam membentuk banyak pohon keputusan untuk meningkatkan akurasi dan meminimalkan overfitting. Memiliki kelebihan lebih akurat daripada satu pohon atau decision tree dan dapat menangani data yang besar dan kompleks. Sedangnya kekurangannya waktu pelatihan dan testing menjadi lebih lambat pada saat prediksi data yang besar. Menjadikan algoritma ini cocok karena memiliki data yang besar dan kompleks.

Pemodelan dilakukan menggunakan library `sklearn.ensemble` dengan memasukkan `X_train_selected` dan `y_train` untuk melatih model, lalu menggunakan `X_test_scaled` dan `y_test` untuk menguji model dengan data testing yang tidak ada pada data training. Parameter awal yang digunakan merupakan `n_estimators=50` dan `random_state=42` menjadikan model berjalan dengan sangat lama dan akurasi kurang memuaskan. Parameter yang digunakan setelah melakukan tuning parameter, yaitu `max_depth = 20` merupakan kedalaman maksimum pohon, dan `n_estimators = 3` jumlah pohon dalam forest.

### 3. **K-Nearest Neighbor**
Merupakan algoritma klasifikasi dengan memanfaatkan data terdekat untuk melakukan prediksi pada data baru yang belum dikenal (data testing). Cocok digunakan untuk jenis data kategorikal maupun numerikal. Kelebihan dari model ini adalah sederhana dan intuitif, cocok untuk data dengan struktur non-linear, dan performa baik pada dataset kecil dan bersih. Sedangkan kekurangannya sensitif terhadap skala dan perlu dilakukan normalisasi dan tidak efektif pada dataset yang besar, akan lambat pada proses prediksi. Menjadikan model ini cocok karena memiliki struktur non-linear.

Pada proyek ini, pemodelan dilakukan dengan menggunakan library `sklearn.neighbors` dengan melatih model dan menguji data dengan menggunakan data testing. Pada awal pemodelan, parameter yang digunakan `n_estimators = 5` dan `metric = 'mikowski'` menjadikan model kurang dapat mengenali data testing. Parameter yang digunakan setelah memasuki proses parameter tuning, yaitu `n_neighbors = 15` merupakan jumlah tetangga yang dipertimbangan, dan `weights = 'uniform` merupakan bobot atau penjarakan untuk kontribusi kepada tetangga.

### 4. Gradient Boosing
Merupakan algoritma yang bekerja membangun beberapa model decision tree secara bertahap di mana setiap model memperbaiki kesalahan dari model sebelumnya. Keuntungan pada model ini adalah akurat untuk data tabular, dan dapat menangani fitur numerik dan kategorikal. Sedangkan kekurangannya lebih lambat dibandingkan dengan random forest, terutama untuk dataset besar dan rentan terhadap overfitting jika tidak disetel dengan baik. Menjadikan cocok dengan data karena memiliki banyak fitur numerikal dan kategorikal

Pada proyek ini, pemodelan dilakukan dengan menggunakan library `sklearn.ensemble`, merupakan library khusus untuk algoritma Gradient Boosting. Parameter yang digunakan pada model merupakan `n_estimators = 50` dan `random_state = 42` menjadikan model tidak bisa sama sekali mengenali data baru. Parameter yang digunakan setelah model ditunning adalah `n_estimators = ` merupakan jumlah pohon boosting, `max-depth = ` merupakan maksimal kedalaman pohon, dan `min-samples-split = ` merupakan jumlah minimal sample yang dibutuhkan untuk membagi (split) sebuah node.

Proses data preparation sangat membantu proses modeling dan setiap langkahnya sangat berkaitan. Penambahan beberapa fitur membantu dalam menambahkan korelasi pada data, label encoding dan normalisasi data membantu data dikenali oleh model dan bekerja dengan lebih cepat. Pelaksanannya dilakukan setelah spliting data membantu model tidak bocor antara training dan testing. Pemilihan fitur melalui teknik Lasso dan mempertimbangkan *confussion matrix* membantu dalam reduksi data, mengurangi kompleksitas model, dan meningkatkan akurasi dengan menghindari data dengan korelasi yang tinggi. Penggunaan teknik ini juga secara langsung menjawab pernyataan permasalahan atas fitur apa saja yang berpengaruh kepada keselamatan dan diagnosis kanker paru-paru.

## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation

Metrik yang digunakan pada proyek ini meliputi:
1. **Accuracy**, presentase prediksi yang benar dari keseluruhan data. Dengan rumus: 
<br>
$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$
</br>
2. **Precision**, presentase seberapa banyak prediksi sebagai positif dan yang benar-benar positif.
<br>$$
\text{Precision} = \frac{TP}{TP + FP}
</br>$$
3. **Recall**, presentase atau nilai data yang benar-benar positif dan yang berhasil terdetaksi oleh model.
<br>$$
\text{Recall} = \frac{TP}{TP + FN}
</br>$$
4. **F1-Score**, rata-rata harmonis dari precision dan recall dan berguna pada saat data tidak seimbang.
<br>$$
\text{F1 Score} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
<\br>$$


Berikut adalah evaluasi pada model:

*Tabel 2, Matriks Evaluasi Model Training*
Model | Accuracy | Precission | Recall | F1 Score
----------|----------|----------|----------|----------
Decision Tree | 0,78 | 0,72 | 0,001 | 0,002
Random Forest | 0,81 | 0,67 | 0,29 | 0,40
KNN | 0,79 | 0,57 | 0,19 | 0,29
Gradient Boosting | 0,78 | 0,00 | 0,00 | 0,00

*Tabel 3, Matriks Evaluasi Model Testing*
Model | Accuracy | Precission | Recall | F1 Score
----------|----------|----------|----------|----------
Decision Tree | 0,78 | 0,19 | 0,001 | 0,002
Random Forest | 0,75 | 0,23 | 0,06 | 0,093
KNN     | 0,73 | 0,21 | 0,07 | 0,10
Gradient Boosting | 0,78 | 0,00 | 0,00 | 0,00

Proses evaluasi pada proyek ini akan menggunakan akurasi, confusion matrix, dan f1-score. Mengingat rasio nilai pada target tidak merata dengan nilai 1 lebih sedikit, maka evaluasi model akan lebih memperhatikan prediksi ke nilai 1 atau true positif (maka dari itu metrik f1-score lebih cocok).

1. **Decision Tree**, Akurasi pada pelatihan dan prediksi data 78%. Dari gambar di bawah, hanya 9 data yang diprediksi sebagai true positive. Diperoleh skor F1 pada prediksi 0,002.
<br>
<img src="https://github.com/ayalya/PredictiveAnalysis_LungCancer/blob/main/asset/12CMDT.png" align="center"><a>

*Gambar13, Confussion Matrix Decision Tree*
</br>
2. **Random Forest**, terdapat 1824 data yang dapat diprediksi sebagai true positif pada model ini. Model ini mendapatkan akurasi pelatihan 0,98 dam akurasi prediksi 0,72 sedangkan nilai F1 pada prediksi 0,14
<br>
<img src="https://github.com/ayalya/PredictiveAnalysis_LungCancer/blob/main/asset/13CMRF.png" align="center"><a>

*Gambar14, Confussion Matrix Random Forest*
</br>
3. **KNN**, dapat memprediksi 1279 data sebagai true positif. Mendapatkan akurasi pelatihan 0,79 dan akurasi prediksi 0,73. Model mendapatkan skor f1 pada pelatihan 0,29 dan prediksi 0,1
<br>
<img src="https://github.com/ayalya/PredictiveAnalysis_LungCancer/blob/main/asset/13CMKNN.png" align="center"><a>

*Gambar15, Confussion Matrix KNN*
</br>
4. **Gradient Boosting**, model ini tidak dapat mengenali data true positif sehingga mendapatkan skor f1 pada pelatihan dan pengujian adalah 0. Namun masih bisa memprediksi nilai true false, memiliki akurasi pelatihan dan pengujian sebesar 78%.
<br>
<img src="https://github.com/ayalya/PredictiveAnalysis_LungCancer/blob/main/asset/14CMGB.png" align="center"><a>

*Gambar16, Confussion Matrix Gradient Boosting*
</br>
Akurasi dan nilai F1 pada keseluruhan model belum menunjukkan hasil yang cukup baik menyentuh angka rata-rata belum menyentuh angka 0,8 dan nilai F1 tidak sampai 0,5. Hal ini dapat terjadi karena:
1. Persebaran pada data kategorikal terlalu menyebar dan model terlalu menggeneralisasi sehingga sulit untuk mengenali data, bahkan overfitting.
2. Meskipun persebaran data merata, target pada proyek ini yaitu `survived` memiliki sebaran yang tidak seimbang atau undersampling. Kasus pasien yang tidak selamat lebih banyak dibandingkan dengan pasien yang hidup.

#### Hasil Evaluasi
Untuk membandingkan seluruh nilai akurasi, berikut adalah perbandingan nilai akurasi pada model:
<img src="https://github.com/ayalya/PredictiveAnalysis_LungCancer/blob/main/asset/15accuracy.png" align="center"><a>

*Gambar17, Confussion Matrix Random Forest*

Berdasarkan gambar perbandingan akurasi di atas, model dengan akurasi tertinggi adalah Random Forest dan KNN. Selain akurasi, nilai F1 pada pelatihan dan prediksi merupakan nilai teringgi dengan jumlah kesalahan prediksi paling minimum dan paling banyak memprediksi nilai true positif, terutama pada model Random Forest.

## Conclusion
Berdasarkan hasil yang diperoleh pada proses EDA dan pengujian model terbaik untuk menentukan kejadian kanker paru-paru dapat menjawab pertanytaan permasalahan dan dapat disimpulkan bahwa:
1. Beberapa fitur yang terbukti paling berpengaruh terhadap diagnosis dan keselamatan pasien kanker paru adalah: `country`, `cancer_stage`, `age_group`, `survival_months`, `bmi`, dan `cholesterol_group`. Fitur country mencerminkan kondisi fasilitas kesehatan yang tersedia, sedangkan stadium kanker (`cancer_stage`) menunjukkan korelasi kuat dengan harapan hidup—semakin tinggi stadiumnya, semakin rendah kemungkinan keselamatan pasien. Hal ini memperkuat pentingnya deteksi dini dan pola hidup sehat.
2. Ditemukan adanya korelasi antara keberlangsungan hidup dengan stadium kanker. Pasien dengan stadium yang lebih rendah cenderung memiliki survival rate yang lebih tinggi. Namun, jenis perawatan (treatment type) tidak menunjukkan hubungan signifikan dengan stadium, menandakan bahwa efektivitas pengobatan bisa sangat bervariasi dan perlu disesuaikan secara individual. Literatur menunjukkan bahwa operasi adalah pilihan utama, tetapi harus dipertimbangkan berdasarkan kondisi spesifik pasien.
3. Dari pengujian terhadap empat model machine learning: Decision Tree, Random Forest, KNN, dan Gradient Boosting, diperoleh dua model paling optimal:
- **Random Forest** menunjukkan performa terbaik dari segi akurasi dan kestabilan prediksi secara keseluruhan.
- **KNN** memberikan nilai F1 Score tertinggi untuk kelas positif, menunjukkan kemampuannya dalam mendeteksi pasien yang memiliki kemungkinan tinggi untuk bertahan hidup.
4. 

## Referensi
<a name="ref1"></a>[1] J. Joseph and L. W. A. Rotty, “Kanker Paru: Laporan Kasus”, MSJ, vol. 2, no. 1, Jul. 2020.

<a name="ref2"></a>[2] Purangga, Tengku Arya, "Deteksi Kanker Paru-Paru Menggunakan Machine Learning dengan Metode Artificial Neural Network dan Decision Tree" (Skripsi), Fakultas Sains dan Teknologi. UIN Syarif Hidayatullah Jakarta, 2024.

<a name="ref3"></a>[3] Mitra Keluarga, "Penyebab Utama Kanker Paru dan Pengobatan yang Tepat. Terapi di Oncology Center!", Mitra Keluarga. Online. Tersedia: https://www.mitrakeluarga.com/artikel/kanker-paru. [Diakses: 17 Mei 2025].

<a name="ref4"></a>[4] Wulandari, Laksmi, "Terapi Target pada Kanker Paru". Surabaya: Universitas Airlangga Press, 2019.

<a name="ref5"></a>[5] "Hari Kanker Sedunia 2025", RS Roemani Muhammadiah. Online. Tersedia: https://rsroemani.com/rv2/hari-kanker-sedunia-2025-united-by-unique. [Diakses: 17 Mei 2025].

[Github](https://github.com/ayalya/PredictiveAnalysis_LungCancer/tree/main)