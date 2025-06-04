# Laporan Proyek Machine Learning - Fadhilah Hafidz Haditama

## Domain Proyek: Lingkungan (Polutan)

Pencemaran udara telah menjadi isu lingkungan yang muncul akibat bertambah luasnya wilayah industrial. Polutan yang dihasilkan dapat berdampak besar, terutama polutan berpartikel kecil, seperti PM2.5. Particulate Matter 2.5 (PM2.5) atau fine particles merupakan partikulat yang berukuran diameter kurang dari 2.5 µm yang sering dikaitkan dengan berbagai penyakit serius, seperti gangguan pernapasan, penyakit kardiovaskular, serta peningkatan risiko kematian dini (Martins dan Da Graca 2018). Jenis polutan ini telah mendapat perhatian intensif selama dua dekade terakhir akibat pengaruhnya terhadap kesehatan.

Konsentrasi PM2.5 sangat dipengaruhi oleh kondisi meteorologis seperti suhu, kelembaban, kecepatan angin, tekanan udara, dan curah hujan (Yang *et al.* 2017). Dalam proyek ini, Pendekatan prediktif dengan model machine learning (ML) dikembangkan dengan mengintegrasikan data konsentrasi polutan PM2.5 dan iklim historis. Model ini bertujuan untuk mengidentifikasi hubungan antara variabel meteorologis dengan lonjakan PM2.5 yang sering terjadi selama periode transisi musim atau saat terjadi gangguan atmosferik. Algoritma seperti Random XGBoost, KNN, dan Decision Tree mampu menangkap pola temporal dan nonlinier dalam data lingkungan. Link dataset dapat diakses [disini](https://www.kaggle.com/datasets/rupakroy/lstm-datasets-multivariate-univariate).

## Business Understanding

### Problem Statements
1. Faktor meteorologi mana yang paling berpengaruh terhadap konsentrasi polutan PM2.5?
2. Bagaimana perbandingan nilai aktual dan prediksi konsentrasi polutan PM2.5 dari model terbaik berdasarkan parameter meteorologi?

### Goals
1. Mengetahui faktor meteorologi yang memengaruhi konsentrasi polutan PM2.5
2. Menentukan model terbaik dan perbandingan nilai aktual dan prediksi konsentrasi polutan PM2.5 berdasarkan parameter meteorologi yang berpengaruh

### Solution Statement
1. Melakukan analisis awal terhadap data PM2.5 dan data meteorologi untuk memahami pola dan hubungan antara faktor cuaca (suhu udara, kelembaban udara, tekanan udara, kecepatan angin, arah angin, curah hujan) dengan konsentrasi PM2.5. Analisis dilakukan menggunakan teknik visualisasi data dan statistik deskriptif untuk mengetahui korelasi antar variabel dan memahami pengaruh masing-masing parameter cuaca terhadap tingkat polusi udara.
2. Menggunakan algoritma machine learning untuk membandingkan performa berbagai model dalam memprediksi konsentrasi PM2.5 berdasarkan faktor-faktor meteorologi. Model yang digunakan terdiri dari empat algoritma machine learning, yaitu Linear Regression, Desicion Tree Regressor, XGBoost Regressor, dan K-Nearest Neighbors (KNN). Algoritma dioptimalkan dengan hyperparameter tuning dengan library optima dan GridSearchCV.
3. Evaluasi performa dilakukan dengan menggunakan Mean Square Error (MSE), Root Mean Square Error (RMSE), Mean Absolute Error (MAE), dan R-squared (R²) untuk masing-masing model.


## Data Understanding

Dataset yang digunakan berasal dari Kaggle dengan Usability 9.12/10. Dataset ini mencatat informasi cuaca dan tingkat polusi udara setiap jam selama lima tahun di Kedutaan Besar Amerika Serikat di Beijing, Tiongkok. Data mencakup informasi waktu (tanggal dan jam), tingkat polusi berupa konsentrasi PM2.5, serta berbagai parameter cuaca seperti:
*   DEWP: Titik embun (Dew Point)
*   TEMP: Suhu udara (Temperature)
*   PRES: Tekanan udara (Pressure)
*   CBWD: Arah angin gabungan (Combined Wind Direction)
*   Iws: Kecepatan angin kumulatif (Cumulated Wind Speed)
*   Is: Jumlah jam kumulatif terjadinya salju (Cumulated Hours of Snow)
*   Ir: Jumlah jam kumulatif terjadinya hujan (Cumulated Hours of Rain)

### Variabel-variabel pada Restaurant UCI dataset adalah sebagai berikut:
Terdapat 9 fitur, yaitu:
1. date : Tanggal dan waktu pengambilan data (%YYYY-%mm-%dd %HH:%MM:%ss)
2. pollution : Konsentrasi polutan PM2.5 (polutan dengan diameter <= 2,5 µm) (µg/m3)
3. dew : Dewpoint atau suhu titik embun, suhu saat udara menjadi jenuh dan uap air mulai mengembun jadi air (°C)
4. temp : Suhu udara (°C)
5. press : Tekanan udara (hPa)
6. wnd_dir : Combined Wind direction atau arah angin gabungan
7. wnd_spd : Jumlah kumulatif wind speed atau kecepatan angin (m/s)
8. snow : Jumlah jam kumulatif terjadinya salju (jam)
9. rain : Jumlah jam kumulatif terjadinya hujan (jam)

| Jumlah Baris | Jumlah Kolom 
| ------ | ------ 
| 43800 | 9 

| Missing Value | Data Duplikat
| ------ | ------ 
| 0 | 0

### Exploratory Data Analysis - Menangani Nilai Tidak Praktis Polutan dan Outliers

#### Menghilangkan data polutan kurang dari 7
![drop data polution kurang dari 7](https://github.com/fadhilahhafidzh/pm2.5-model/blob/main/Gambar/drop%20data%20polution%20kurang%20dari%207.png)

Pembersihan data dengan batas nilai 7 dikarenakan niai minimum polutan di US Embassy, beijing, China berniai 7. Setelah nilai bawahnya dihilangkan sebanyak 2842 baris, jumlah baris pada dataframe yang telah dibersihkan mencapai 40958.

#### Pembersihan outliers
![cleaning outliers](https://github.com/fadhilahhafidzh/pm2.5-model/blob/main/Gambar/cleaning%20outliers.png)

#### Deskripsi data statistik
![statistik data](https://github.com/fadhilahhafidzh/pm2.5-model/blob/main/Gambar/statistik%20data.png)

Fungsi describe() menyatakan:
- Count = jumlah sampel data.
- Mean = nilai rata-rata kolom.
- Std = standar deviasi kolom.
- Min = nilai minimum setiap kolom.
- 25% = kuartil pertama dari nilai kolom.
- 50% = kuartil kedua atau median (nilai tengah) dari nilai kolom.
- 75% = kuartil ketiga dari nilai kolom.
- Max = nilai maksimum dari nilai kolom.

Statistik data menunjukkan rata-rata rentang polutan tinggi hingga mencapai 88,3 µg/m3 dengan rentang dewpoint -37-28 °C, suhu udara -19-42 °C, tekanan udara +- 1000 hPa, kecepatan angin kumulatif 0,45-565,5 m/s, jam kumulatif bersalju mencapai 27 jam, jam kumulatif hujan mencapai 36 jam.

### Exploratory Data Analysis - Univariet Analysis
Pembagian data kategorik dan numerik
![pembagian data num dan cat](https://github.com/fadhilahhafidzh/pm2.5-model/blob/main/Gambar/pembagian%20data%20num%20dan%20cat.png)
7 fitur numerik dan 1 fitur kategorik.

#### Data Kategorikal
![exploratory kategori](https://github.com/fadhilahhafidzh/pm2.5-model/blob/main/Gambar/exploratory%20kategori.png)
Arah angin menunjukkan dominasi dari arah SE, disusul NW, cv, dan NE.
Keterangan:
- SE = southeast atau tenggara
-NW = northwest atau barat-laut
- cv = calm and variable, arah angin tidak dapat ditentukan dengan jelas karena atmosfer yang tenang. Muncul ketika terjadi peralihan arah angin.
- NE = northeast atau timur-laut

#### Data Numerik
![exploratory numerik](https://github.com/fadhilahhafidzh/pm2.5-model/blob/main/Gambar/exploratory%20numerik.png)
Berdasarkan fitur target 'pollution' diketahui bahwa:
- Fitur target memiliki data terdistribusi banyak dibawah 50 µg/m3 dan mendekati 300 µg/m3.
- Rentang konsentrasi polutan cukup tinggi dari 0 hingga 300 µg/m3. Hal ini menunjukkan nilai rata-rata perjam sangat variatif dan berkemungkinan diakibatkan faktor cuaca per jam yang bervariasi.
- Distribusi konsentrasi polutan miring ke kanan (right-skewed). Hal ini akan berimplikasi pada model.

### Exploratory Data Analysis - Multivariet Analysis
Konsentrasi polutan - arah angin
![multivariet kategori](https://github.com/fadhilahhafidzh/pm2.5-model/blob/main/Gambar/multivariet%20kategori.png)

Pada saat angin tenang dan mengarah ke tenggara konsentrasi PM2.5 lebih tinggi dibandingkan ketika angin menuju barat-laut maupun timur-laut.

### Konsentrasi polutan - data numerik
![pairplot](https://github.com/fadhilahhafidzh/pm2.5-model/blob/main/Gambar/pairplot.png)

Fungsi pairplot menampilkan scatter plot untuk setiap pasangan variabel numerik untuk menunjukkan distribusi masing-masing variabel. Fungsi utama pairplot adalah untuk eksplorasi data secara visual agar kita dapat mengidentifikasi korelasi, pola, outlier, atau distribusi dari variabel-variabel yang ada.

Pada gambar pairplot tersebut, kita bisa fokus pada baris pertama yang merepresentasikan hubungan variabel pollution dengan variabel lainnya. Dari baris ini, kita bisa lihat bahwa:
- Hubungan antara pollution dan dew tampak menyebar acak denga sedikit naik.
- Hubungan pollution dan temp tersebar acak.
- Korelasi antara pollution dan press terlihat tidak kuat, dengan titik-titik yang menyebar luas tanpa pola linier yang jelas.
- Hubungan terhadap wind speed, snow, dan rain terlihat berbanding terbalik. Hal ini bisa menjadi indikasi bahwa kondisi angin rendah, hujan dan salju minim bisa berkontribusi pada penumpukan polusi udara.

### Korelasi Antarfitur

![matriks korelasi](https://github.com/fadhilahhafidzh/pm2.5-model/blob/main/Gambar/matriks%20korelasi.png)

Korelasi adalah ukuran statistik yang menunjukkan sejauh mana hubungan linier antara dua variabel. Nilai korelasi berkisar dari -1 hingga 1, di mana 1 menunjukkan hubungan positif sempurna, -1 menunjukkan hubungan negatif sempurna, dan 0 berarti tidak ada hubungan linier. Fungsi korelasi adalah untuk mengidentifikasi apakah dan seberapa kuat dua variabel berubah secara bersamaan.

Korelasi kuat ditunjukkan antara variabel, yaitu:
1. dew dan temp (r=0,83)→ Korelasi positif kuat: dewpoint meningkat, suhu udara juga ikut meningkat
2. temp dan press (r=-0,83)
→ Korelasi negatif kuat: semakin tinggi suhu udara, tekanan udara cenderung menurun.
3. dew dan press (r = -0.78)
→ Korelasi negatif kuat: titik embun tinggi berkaitan dengan tekanan yang lebih rendah.

Nilai korelasi lainnya menunjukkan korelasi lemah hingga tidak menunjukkan korelasi sama sekali.

## Data Preparation

### Drop Kolom Korelasi Rendah (rain, snow)

![drop korelasi rendah](https://github.com/fadhilahhafidzh/pm2.5-model/blob/main/Gambar/drop%20korelasi%20rendah.png)

### Encoding Fitur Kategori (Wind Direction)
![encoding](https://github.com/fadhilahhafidzh/pm2.5-model/blob/main/Gambar/encoding.png)

### Split-Train-Test
![split data](https://github.com/fadhilahhafidzh/pm2.5-model/blob/main/Gambar/split%20data.png)

Total data sampel untuk train, yaitu 32766, sedangkan data test nya 8192. Pembagian ini didasarkan perbandingan train-test 80:20.

### Normalisasi
Normalisasi adalah proses mengubah skala atau distribusi nilai data agar berada dalam rentang tertentu atau memiliki karakteristik statistik tertentu, tanpa mengubah informasi dasarnya. Tujuannya adalah agar data dari berbagai sumber atau skala bisa dibandingkan secara adil atau digunakan secara efektif dalam analisis atau model.

#### Normalisasi data train

![standarisasi data train](https://github.com/fadhilahhafidzh/pm2.5-model/blob/main/Gambar/standarisasi%20data%20train.png)

#### Normalisasi data test

![standarisasi data test](https://github.com/fadhilahhafidzh/pm2.5-model/blob/main/Gambar/standarisasi%20data%20test.png)

## Modeling
1. Linear Regression
   
Linear Regression adalah metode statistik dan machine learning untuk memodelkan hubungan linear antara satu atau lebih variabel input (fitur) dengan variabel output (target). Model ini mencari garis atau hyperplane terbaik yang meminimalkan jarak antara prediksi dan data sebenarnya (dengan cara meminimalkan residual/error).

Pada kode tersebut, LinearRegression diimpor dari scikit-learn untuk membuat objek model regresi linear dengan parameter default, yaitu fit_intercept=True, copy_X=True, n_jobs=None, dan positive=False. Kemudian, model tersebut dilatih dengan data pelatihan menggunakan fit(X_train, y_train), sehingga model mempelajari pola dari data tersebut. Setelah itu, model yang sudah dilatih digunakan untuk memprediksi nilai target pada data uji X_test dengan predict(X_test), dan hasil prediksi disimpan dalam variabel pred_LR.

2. XGBoost Regressor
   
XGBoost Regressor adalah algoritma machine learning berbasis boosting yang dirancang khusus untuk masalah regresi dengan tujuan meningkatkan akurasi prediksi melalui penggabungan banyak model pohon keputusan yang lemah secara iteratif. Algoritma ini membangun model secara bertahap dengan memperbaiki kesalahan prediksi dari model sebelumnya, sehingga menghasilkan model yang kuat dan tahan terhadap overfitting.

Dalam pengolahan data, data fitur dan target terlebih dahulu dipersiapkan dan dibagi menjadi data latih dan data uji dalam numpy array. Kemudian, fungsi objective yang digunakan oleh Optuna untuk melakukan pencarian hyperparameter terbaik. Fungsi ini menerima objek trial yang bertugas memberikan nilai hyperparameter secara otomatis dalam ruang pencarian. Pemilihan hyperparameter untuk model XGBoost secara otomatis oleh Optuna. Hasilnya, max_depth=6, n_estimators=123, subsample=0.7552836445320749, colsample_bytree=0.9164527190427988, random_state=780, learning_rate=0.06146493598489172, n_jobs=-1. Selanjutnya, membuat studi Optuna dengan tujuan minimasi nilai MSE, lalu menjalankan pencarian hyperparameter sebanyak 250 percobaan (trial) dengan memanggil fungsi objective yang sudah didefinisikan. Hasil digunakan untuk membuat ulang dan melatih kembali model XGBoost dengan hyperparameter terbaik yang sudah ditemukan oleh Optuna.

3. Decision Tree
   
Decision tree adalah algoritma machine learning yang digunakan untuk klasifikasi dan regresi dengan cara membagi data secara berulang berdasarkan fitur-fitur tertentu untuk membentuk sebuah pohon keputusan. Algoritma ini bekerja dengan membuat serangkaian pertanyaan “ya” atau “tidak” yang membelah data ke dalam cabang-cabang hingga mencapai daun (node terminal) yang berisi prediksi output. Setiap percabangan dipilih berdasarkan kriteria yang memaksimalkan pemisahan data, seperti pengurangan impuritas (misalnya Gini atau Entropy untuk klasifikasi, atau variansi untuk regresi). Decision tree mudah dipahami dan diinterpretasi karena hasilnya berupa aturan keputusan yang jelas, namun bisa rentan terhadap overfitting jika pohon terlalu dalam.

Kode ini menggunakan Optuna untuk melakukan tuning hyperparameter pada model Decision Tree Regressor dengan tujuan meminimalkan Mean Squared Error (MSE) pada data uji. Fungsi objective mendefinisikan ruang pencarian hyperparameter seperti kedalaman pohon (max_depth), jumlah minimum sampel untuk membagi node (min_samples_split), jumlah minimum sampel pada daun (min_samples_leaf), dan jumlah fitur yang dipakai untuk membagi node (max_features). Setelah menjalankan 100 percobaan, Optuna menemukan konfigurasi terbaik dengan max_depth 11, min_samples_split 47, min_samples_leaf 25, dan max_features menggunakan metode 'None', yang menghasilkan MSE sebesar 4079,7. Dengan pengaturan ini, model diharapkan memberikan prediksi yang lebih akurat dan stabil pada data uji.

5. K-Nearest Neighbors (KNN)
   
K-Nearest Neighbors (KNN) adalah algoritma machine learning sederhana yang digunakan untuk klasifikasi maupun regresi dengan cara mencari sejumlah tetangga terdekat (k) dari data baru berdasarkan jarak (misalnya Euclidean) ke data latih, lalu memprediksi output berdasarkan nilai tetangga tersebut. Proses tuning hyperparameter KNN biasanya melibatkan pemilihan nilai k terbaik yang dapat dilakukan dengan metode seperti GridSearchCV, yang secara otomatis mencoba berbagai nilai k dan menilai performa model menggunakan teknik cross-validation. Pada contoh yang diberikan, tuning menghasilkan nilai k terbaik sebesar 9, yang berarti model menggunakan 9 tetangga terdekat untuk membuat prediksi. Kelebihan KNN adalah mudah dipahami dan diimplementasikan, serta fleksibel karena tidak memerlukan asumsi distribusi data. Namun, KNN juga memiliki kekurangan seperti kinerjanya yang menurun pada data berdimensi tinggi (curse of dimensionality), serta kebutuhan komputasi yang besar saat dataset sangat besar karena harus menghitung jarak ke seluruh data latih saat prediksi.

## Evaluation
Evaluasi yang digunakan projek ini, yaitu Mean Square Error (MSE), Root Mean Square Error (RMSE), Mean Absolute Error (MAE), dan R-squared (R²) untuk masing-masing model.

Dalam proyek ini digunakan empat metrik evaluasi utama untuk menilai kinerja model regresi:
- Mean Squared Error (MSE): Mengukur rata-rata kuadrat selisih antara nilai aktual dan nilai prediksi. Semakin kecil nilainya, semakin baik prediksi model.

![MSE](https://github.com/fadhilahhafidzh/pm2.5-model/blob/main/Gambar/MSE.png)

- Root Mean Squared Error (RMSE): Akar dari MSE, yang mengembalikan kesalahan dalam satuan yang sama dengan data asli. Cocok untuk menilai seberapa jauh prediksi model dari nilai aktual.
  
![RMSE](https://github.com/fadhilahhafidzh/pm2.5-model/blob/main/Gambar/RMSE.png)

- Mean Absolute Error (MAE): Menghitung rata-rata selisih absolut antara nilai aktual dan prediksi. Tidak terlalu sensitif terhadap outlier, berbeda dengan MSE.
  
![MAE](https://github.com/fadhilahhafidzh/pm2.5-model/blob/main/Gambar/MAE.png)

- R-squared (R²): Menjelaskan proporsi variansi dari target yang dapat dijelaskan oleh fitur. Nilai R² berkisar dari 0 hingga 1 (atau bisa negatif jika model buruk). Nilai mendekati 1 berarti model menjelaskan sebagian besar variasi data.
  
![r square](https://github.com/fadhilahhafidzh/pm2.5-model/blob/main/Gambar/r%20square.png)

Setiap model dalam proyek ini dibandingkan menggunakan keempat metrik di atas. Misalnya:
- Model dengan MSE dan RMSE paling kecil dianggap paling akurat secara keseluruhan.
- Model dengan MAE rendah menunjukkan kesalahan prediksi rata-rata yang kecil.
- Model dengan R² mendekati 1 dianggap paling baik dalam menjelaskan hubungan antara input dan output.

Jika, misalnya, model XGBoost memiliki MSE dan RMSE yang paling rendah serta R² yang tinggi dibanding Decision Tree atau KNN, maka dapat disimpulkan bahwa XGBoost paling baik dalam memodelkan data dalam proyek ini.

### Hasil Evaluasi
1. Nilai RMSE, MSE, MAE, R² seluruh model
   
![hasil evaluasi](https://github.com/fadhilahhafidzh/pm2.5-model/blob/main/Gambar/hasil%20evaluasi.png)

XGBoost Regressor adalah model terbaik secara keseluruhan karena memiliki MAE, MSE, dan RMSE paling rendah. Artinya, kesalahan prediksinya paling kecil.
R² juga bernilai tertinggi (0.4037). Model ini mampu menjelaskan ~40.4% variasi dalam data. Meskipun R² = 0.4037 tergolong rendah secara umum, XGBoost tetap menjadi model terbaik di antara yang dibandingkan karena memiliki kesalahan prediksi terendah. Ini menunjukkan bahwa model sudah menangkap sebagian pola dalam data, tetapi masih banyak variabilitas yang belum terjelaskan. Perlu eksplorasi lebih lanjut terhadap fitur tambahan dan transformasi data.

2. Fitur yang paling berpengaruh
   
![fitur berpengaruh](https://github.com/fadhilahhafidzh/pm2.5-model/blob/main/Gambar/fitur%20berpengaruh.png)

Berdasarkan hasil pemodelan menggunakan XGBoost Regressor untuk memprediksi konsentrasi PM2.5, fitur yang paling berpengaruh adalah arah angin dari barat laut (wnd_dir_NW), titik embun (dew), dan arah angin dari tenggara (wnd_dir_SE). Ketiga fitur ini memiliki nilai importance tertinggi, yang menunjukkan bahwa arah angin dominan dan parameter kelembapan udara berperan besar dalam pergerakan serta konsentrasi PM2.5 di atmosfer. Sementara itu, fitur seperti tekanan udara (press) dan variasi arah angin (wnd_dir_cv) memiliki pengaruh yang lebih kecil terhadap hasil prediksi, kemungkinan karena kontribusinya lebih rendah dalam menjelaskan variasi data PM2.5 secara signifikan. Temuan ini menegaskan bahwa dalam konteks lokal data tersebut, faktor meteorologi terkait arah dan kelembapan angin menjadi kunci utama dalam memodelkan polusi udara.

3. Perbandingan hasil prediksi dan aktual
   
![prediksi vs aktual](https://github.com/fadhilahhafidzh/pm2.5-model/blob/main/Gambar/prediksi%20vs%20aktual.png)

Hasil prediksi model XGBoost Regressor terhadap konsentrasi PM2.5 menunjukkan bahwa meskipun terdapat korelasi umum antara nilai prediksi dan nilai asli, model cenderung kurang akurat pada nilai PM2.5 yang tinggi, ditandai dengan banyaknya titik yang berada di bawah garis referensi (y = x). Ini mengindikasikan bahwa model sering melakukan underprediction saat PM2.5 berada dalam kondisi ekstrem atau tinggi, kemungkinan karena dominasi data pada rentang rendah-menengah dan minimnya representasi kasus ekstrem dalam data latih..

## Kesimpulan
1. Analisis fitur importance model XGBoost Regressor diketahui bahwa arah angin dari barat laut (wnd_dir_NW), titik embun (dew), dan arah angin dari tenggara (wnd_dir_SE) adalah faktor meteorologi yang paling berpengaruh terhadap konsentrasi PM2.5. Hal ini menunjukkan bahwa pola angin dominan dan kelembapan udara memiliki peran penting dalam mengendalikan pergerakan dan akumulasi polutan di atmosfer. Pengetahuan ini dapat menjadi dasar bagi pengambil kebijakan dalam menyusun strategi mitigasi yang lebih tepat sasaran, misalnya dengan mengintegrasikan informasi arah angin ke dalam sistem peringatan dini polusi udara.

2. Model XGBoost Regressor juga berhasil memberikan prediksi nilai PM2.5 berdasarkan data meteorologi dan menunjukkan performa terbaik dibandingkan model lain. Nilai evaluasi yang dicapai adalah MAE, MSE, dan RMSE terendah, serta nilai R² sebesar 0.4037, yang berarti model mampu menjelaskan sekitar 40,4% variasi dalam data. Perbandingan antara nilai aktual dan prediksi menunjukkan bahwa model cukup akurat untuk nilai PM2.5 dalam rentang rendah-menengah, tetapi cenderung _underpredict_ pada nilai ekstrem. Ini menunjukkan bahwa model belum optimal untuk mendeteksi lonjakan polusi secara presisi, namun bisa menangkap pola umum. Perbaikan dapat dilakukan dengan menambahkan fitur diluar faktor meteorologi, seperti waktu (jam, musim), lag data, dan variabel aktivitas manusia (industri dan transportasi) agar model mampu menangkap dinamika yang lebih kompleks.

## Referensi

Martins NR, Da Graca GC. 2018. Impact of PM2. 5 in indoor urban environments: A review. *Sustainable Cities and Society*. 42(1):259-275.

Yang Q, Yuan Q, Li T, Shen H, Zhang L. 2017. The relationships between PM2. 5 and meteorological factors in China: seasonal and regional variations. *International journal of environmental research and public health*. 14(12):1-19.
