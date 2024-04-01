# Laporan Proyek Machine Learning - Muhammad Edya Rosadi
## Domain Proyek
Tujuan utama proyek ini adalah untuk menganalisis dan memahami faktor-faktor yang mempengaruhi penghasilan individu dan untuk mengembangkan model prediktif yang mampu membuat prediksi yang akurat. Dalam proyek ini, digunakan dataset Adult/Census Dataset yang bersumber dari UCI Machine Learning Repository. Dataset ini merupakan kumpulan data yang dirancang untuk memprediksi apakah penghasilan individu melebihi $50.000 per tahun berdasarkan sensus dari tahun 1994. Proyek ini tidak hanya berfokus pada pembuatan model prediktif yang akurat tetapi juga memberikan gambaran tentang faktor-faktor yang mempengaruhi penghasilan individu, yang dapat bermanfaat bagi pembuat kebijakan, perusahaan, dan individu dalam memahami dinamika pasar kerja.
 
**Rubrik/Kriteria Tambahan:**
-  Dataset ini dipilih karena menawarkan tantangan dalam pemodelan data kategorikal dan numerik, serta penanganan missing values. Ini juga memberikan kesempatan untuk menerapkan teknik pra-pemrosesan data, seperti normalisasi, one-hot encoding, dan imputasi nilai hilang, yang penting dalam pembelajaran mesin. 
-  Untuk itu di proyek ini dibandingkan performa dari berbagai algoritma machine learning, termasuk Logistic Regression, Decision Trees, Random Forests, Support Vector Machines (SVM), dan XGBoost, dalam konteks prediksi kategori penghasilan. Penggunaan berbagai metode evaluasi model, seperti akurasi dan laporan klasifikasi, serta penggunaan Grid Search untuk tuning hyperparameter, merupakan bagian penting dari proyek ini untuk memastikan bahwa model yang dikembangkan memiliki kinerja optimal.
- Beberapa hasil riset atau referensi terkait, antara lain:
    1. Statistical Approach to Adult Census Income Level Prediction - Navoneel Chakrabarty; Sanket Biswas [1]
    2. Retiring Adult: New Datasets for Fair Machine Learning - Frances Ding; Moritz Hardt; John Miller; Ludwig Schmidt [2]
    3. An Investigation into the Prediction of Annual Income Levels Through the Utilization of Demographic Features Employing the Modified UCI Adult Dataset - Md Aminul Islam; Anindya Nag; Nilanjana Roy; Arpita Rani Dey; SM Firoz Ahmed Fahim; Arjan Ghosh [3]

## Business Understanding

**Problem Statements**
Dalam konteks dataset yang digunakan, terdapat beberapa permasalahan seperti:
- Dataset mengandung missing values yang dapat mempengaruhi model pembelajaran mesin.
- Imbalance data dalam klasifikasi pendapatan, yang dapat menyebabkan bias dalam model.
- Kebutuhan preprocessing yang intensif untuk fitur kategorikal dan numerik.
- Kesulitan dalam feature selection dan engineering, yang esensial untuk performa model.
- Scaling issues yang memerlukan normalisasi data untuk memastikan model bekerja dengan efektif.

**Goals**
Tujuan utama proyek ini adalah:
- Menemukan model prediktif yang akurat untuk mengklasifikasikan individu berdasarkan apakah mereka memiliki pendapatan lebih dari $50K per tahun.
- Memastikan model yang dikembangkan adalah fair dan unbiased terhadap kategori minoritas.
- Mengatasi masalah preprocessing untuk meningkatkan kualitas data yang digunakan oleh model.
- Menghilangkan fitur yang kurang berpengaruh terhadap pendapatan individu melalui feature selection.

**Rubrik/Kriteria Tambahan:**
Untuk mencapai tujuan di atas, dua solution statement diusulkan:
1. Menggunakan Berbagai Algoritma Pembelajaran Mesin: Mengadopsi pendekatan ensemble dengan menggunakan beberapa algoritma klasifikasi seperti Logistic Regression, Decision Tree, Random Forest, XGBoost, dan SVM.
2. Hyperparameter Tuning pada Model Baseline: Setelah menentukan satu atau beberapa model dengan performa terbaik, dilanjutkan dengan proses hyperparameter tuning menggunakan teknik seperti GridSearchCV.

## Data Understanding
Pada proyek ini digunakan dataset yang berkaitan dengan informasi pendapatan individu untuk klasifikasi apakah pendapatan seseorang melebihi $50K per tahun atau tidak. Dataset berasal dari repositori UCI Machine Learning ([Adult/Census Income Dataset]) dan dapat digunakan sebagai kasus studi untuk berbagai algoritma klasifikasi dalam pembelajaran mesin. Dataset ini mencakup 14 fitur seperti usia, jenis pekerjaan, pendidikan, status pernikahan, ras, jenis kelamin, jam kerja per minggu, dan negara asal dengan jumlah data sebanyak 48.842.

**Fitur-fitur pada Adult/Census Income UCI dataset adalah sebagai berikut:**
| Fitur | Deskripsi |
| ------ | ------ |
|age|Usia individu|
|workclass|Klasifikasi pekerjaan (misalnya, Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked)|
|fnlwgt|Final weight, jumlah orang yang survei ini representasikan|
|education|Tingkat pendidikan tertinggi yang dicapai (misalnya, Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool)|
|education.num|Jumlah tahun pendidikan yang telah diselesaikan|
|marital.status|Status pernikahan (misalnya, Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse)|
|occupation|Pekerjaan individu (misalnya, Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces)|
|relationship|Hubungan dalam keluarga (misalnya, Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried)|
|race|Ras (misalnya, White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black)|
|sex|Jenis kelamin (Male, Female)|
|capital.gain|Pendapatan kapital|
|capital.loss|Kerugian kapital|
|hours.per.week|Jumlah jam kerja per minggu|
|native.country|Negara asal (misalnya, United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands)|
|income|Pendapatan tahunan, dikategorikan menjadi '<=50K' atau '>50K'|

**Rubrik/Kriteria Tambahan:**
Di dalam proyek ini digunakan exploratory data analysis dan beberapa teknik visualisasi data seperti:
1. Menampilkan Beberapa Baris Pertama dan Terakhir
    Menggunakan df.head(5) dan df.tail(5) untuk menampilkan beberapa baris pertama dan terakhir dari dataset. Ini memberikan gambaran awal tentang struktur data, jenis data di setiap kolom, dan bagaimana data tersebut terlihat.
2. Menghitung Jumlah Entri untuk Setiap Kolom
    df.count() digunakan untuk menghitung jumlah entri non-NA/null di setiap kolom. Ini membantu mengidentifikasi kolom yang mungkin memiliki nilai yang hilang.
3. Deskripsi Statistik Fitur Numerik
    df[numerical_features].describe() memberikan deskripsi statistik (count, mean, std, min, quartiles, max) untuk fitur bertipe numerik. Ini berguna untuk mendapatkan pemahaman tentang distribusi setiap fitur numerik.
4. Visualisasi Fitur Bertipe Numerik
    Menggunakan sns.histplot dan sns.boxplot untuk masing-masing fitur numerik. Histogram dengan KDE (Kernel Density Estimate) menunjukkan distribusi data, sementara boxplot menyediakan visualisasi tentang kuartil, median, dan outliers. Ini penting untuk mengidentifikasi distribusi data dan potensi outliers.
5. Menghitung dan Memvisualisasikan Korelasi Antara Fitur Numerik
    Menggunakan corr_matrix = df[numerical_features].corr() untuk menghitung korelasi antar fitur numerik. Visualisasi heatmap korelasi dengan sns.heatmap membantu mengidentifikasi hubungan antar fitur, yang bisa menjadi informasi penting dalam feature selection dan untuk menghindari multicollinearity dalam model.
6. Analisis Fitur Bertipe Kategori
    Melakukan iterasi melalui categorical_features dan menggunakan value_counts() untuk mengeksplorasi distribusi nilai dalam fitur kategori. Ini memberikan insight tentang kategori yang paling sering muncul dan distribusi kategori.
7. Visualisasi Fitur Bertipe Kategori
    Menggunakan sns.countplot untuk visualisasi frekuensi dari nilai-nilai kategori. Grafik ini membantu dalam mengidentifikasi kategori yang dominan dan distribusi keseluruhan dari fitur kategori.

## Data Preparation
Dalam proses data preparation yang dilakukan mencakup serangkaian langkah penting untuk meningkatkan kualitas dan efektivitas data. Teknik pertama yang diterapkan adalah penggantian nilai yang hilang, di mana nilai '?' diganti dengan np.nan untuk memudahkan identifikasi dan imputasi nilai yang hilang. Imputasi nilai yang hilang dilakukan dengan mengganti nilai kosong pada kolom seperti workclass dan occupation dengan 'Unknown', serta mengisi kolom native.country dengan modusnya. Selain itu, teknik handling data kategorikal dilakukan dengan mengubah nama negara dengan frekuensi di bawah ambang batas tertentu menjadi 'Other', mengurangi dimensionalitas dan meningkatkan kinerja model.

Langkah selanjutnya melibatkan one-hot encoding untuk fitur kategorikal dan standard scaling untuk fitur numerik. Penerapan ColumnTransformer memungkinkan transformasi kolom yang berbeda secara efisien dalam satu langkah, termasuk scaling dan encoding. Dataset kemudian dibagi menjadi set pelatihan dan pengujian untuk validasi model, sementara penghapusan fitur seperti fnlwgt dan target income dari set fitur dilakukan untuk memfokuskan model pada variabel yang relevan. Akhirnya, label encoding diterapkan pada target income untuk mengkonversi nilai kategorikal menjadi numerik.

**Rubrik/Kriteria Tambahan:**
Beberapa teknik data preparation dilakukan dalam proyek ini dengan tujuan untuk mempersiapkan dataset sebelum membangun model pembelajaran mesin. Teknik-teknik ini mencakup:
1. Penggantian Nilai yang Hilang
Mengganti nilai '?' dengan np.nan untuk membuat penanganan nilai yang hilang lebih konsisten. Ini memudahkan proses identifikasi dan imputasi nilai yang hilang.
2. Imputasi Nilai yang Hilang
Melakukan imputasi pada kolom workclass dan occupation dengan mengganti nilai yang hilang dengan 'Unknown'. Ini membantu dalam mempertahankan entri data tanpa menghapus baris yang memiliki informasi penting lainnya.
Untuk kolom native.country, nilai yang hilang diisi dengan modus (nilai yang paling sering muncul). Ini merupakan pendekatan umum untuk fitur kategori dengan asumsi bahwa nilai yang paling sering muncul dapat mewakili nilai yang hilang dengan cukup baik.
3. Handling Categorical Data
Mengubah nama negara dalam fitur native.country dengan frekuensi di bawah ambang batas tertentu menjadi 'Other'. Teknik ini membantu dalam mengurangi dimensionalitas data kategorikal dan meningkatkan kinerja model dengan mengurangi keragaman nilai yang kurang representatif.
4. One-Hot Encoding
Menggunakan OneHotEncoder untuk mengubah fitur kategorikal menjadi format yang dapat diterima oleh algoritma pembelajaran mesin. Ini menghindari ordinality yang tidak diinginkan dalam data kategorikal yang tidak memiliki urutan.
5. Standard Scaling
Menggunakan StandardScaler untuk fitur numerik. Ini mengubah data sehingga memiliki mean nol dan standar deviasi satu. Scaling penting untuk beberapa algoritma pembelajaran mesin yang sensitif terhadap skala fitur, seperti SVM dan k-nearest neighbors.
6. Column Transformer
Menggunakan ColumnTransformer untuk menerapkan transformasi yang berbeda (seperti scaling dan encoding) ke kolom yang berbeda dari dataset dalam satu langkah. Ini memastikan bahwa preprocessing data dilakukan secara efisien dan efektif.
7. Pembagian Dataset
Membagi dataset menjadi set pelatihan dan pengujian menggunakan train_test_split. Ini memungkinkan model untuk dilatih pada satu subset data dan divalidasi pada subset yang lain, yang membantu dalam mengukur kinerja model pada data yang belum pernah dilihat sebelumnya.
8. Penghapusan Fitur
Menghapus kolom fnlwgt dan target income dari fitur yang digunakan untuk pelatihan. Penghapusan fitur berdasarkan pengetahuan domain atau karena fitur tersebut tidak relevan untuk prediksi.
9. Label Encoding untuk Target
Mengubah label target income dari bentuk kategorikal menjadi bentuk numerik (0 dan 1) untuk memudahkan pemodelan.

## Modeling
Di bawah ini adalah model-model yang digunakan dan tahapannya:
- Logistic Regression: Menggunakan parameter max_iter=1000 untuk menentukan jumlah iterasi maksimum pada proses pelatihan.
- Decision Tree Classifier: Model ini dijalankan dengan parameter default, membiarkan model menyesuaikan kedalaman pohon dan parameter lainnya berdasarkan data.
- Random Forest Classifier: Sama seperti Decision Tree, namun menggabungkan prediksi dari banyak pohon keputusan untuk meningkatkan akurasi dan mengurangi overfitting.
- XGBoost: Menggunakan XGBClassifier dengan use_label_encoder=False untuk menghindari peringatan deprecation, dan eval_metric='logloss' untuk mengevaluasi kinerja model selama pelatihan.
- SVM (Support Vector Machine): Digunakan untuk klasifikasi dengan SVC(probability=True) memungkinkan estimasi probabilitas, yang memerlukan pelatihan internal tambahan untuk memperkirakan probabilitas.

Setelah evaluasi awal, tuning hyperparameter dilakukan pada model terpilih (XGBoost, Random Forest, dan SVM) menggunakan GridSearchCV. Proses ini mencari kombinasi parameter terbaik yang menghasilkan akurasi tertinggi.
- XGBoost: Grid search dilakukan pada n_estimators, max_depth, dan learning_rate.
- Random Forest: Grid search meliputi n_estimators, max_depth, dan min_samples_split.
- SVM: Menyesuaikan parameter C, gamma, dan kernel.

**Rubrik/Kriteria Tambahan:**
Berikut tabel kelebihan dan kekurangan setiap algoritma yang digunakan:
| Algoritma | Kelebihan | Kekurangan |
| ------ | ------ | ------ |
| Logistic Regression | - Sederhana dan mudah untuk diimplementasikan.<br> - Cepat dalam pelatihan dan prediksi.<br>- Memiliki interpretasi yang baik. |- Kurang efektif pada ruang fitur yang sangat besar atau dataset yang sangat kompleks.<br>- Rentan terhadap overfitting pada dataset dengan fitur yang sangat banyak. |
| Decision Tree |- Mudah untuk diinterpretasikan dan dijelaskan.<br>- Dapat menangani data kategorikal dan numerik.<br>- Tidak memerlukan normalisasi data. | - Rentan terhadap overfitting, terutama pada pohon yang sangat dalam.<br>- Varians yang tinggi dapat menyebabkan perubahan besar pada struktur pohon dengan perubahan kecil pada data. |
| Random Forest | - Mengurangi overfitting melalui ensemble pohon.<br>- Fleksibel dan dapat digunakan untuk klasifikasi dan regresi.<br>- Performa yang baik pada banyak masalah. | - Lebih lambat dalam pelatihan dan prediksi dibandingkan dengan model yang lebih sederhana.<br>- Lebih sulit untuk diinterpretasikan. |
| XGBoost | - Optimisasi untuk komputasi yang cepat dan penggunaan memori yang efisien.<br>- Mendukung regularisasi untuk mengurangi overfitting.<br>- Fleksibel dan dapat menyesuaikan banyak masalah data. | - Meskipun relatif cepat, bisa menjadi sumber daya yang intensif dan memakan waktu pada dataset sangat besar.<br>- Memiliki kurva belajar yang lebih curam karena banyaknya parameter yang dapat dituning. |
| Support Vector Machine (SVM) | - Efektif dalam ruang dimensi tinggi.<br>- Efektif pada kasus di mana jumlah dimensi lebih besar dari jumlah sampel.<br>- Memiliki keflexibelan dalam pemilihan fungsi kernel. | - Memerlukan pemilihan kernel yang tepat.<br>- Rentan terhadap overfitting pada fitur noise yang banyak.<br>- Waktu dan sumber daya komputasi yang tinggi untuk dataset besar. |

## Evaluation
Berdasarkan analisis terhadap laporan klasifikasi dari berbagai model machine learning, dapat ditarik beberapa kesimpulan mengenai performa masing-masing model dalam tugas klasifikasi penghasilan menggunakan dataset Adult/Census Dataset. Model XGBoost menunjukkan performa terbaik dengan akurasi tertinggi sebesar 87.06%, diikuti oleh Random Forest dengan akurasi 84.65%, SVM dengan 85.23%, dan Decision Tree dengan 82.30%. XGBoost unggul bukan hanya dalam hal akurasi tetapi juga dalam aspek precision dan F1-score, khususnya untuk kelas penghasilan yang lebih tinggi (>50K).

Dalam aspek precision dan recall untuk kelas dengan penghasilan lebih rendah (<=50K), semua model cenderung menampilkan hasil yang kuat, dengan nilai recall yang sangat tinggi yang menunjukkan kemampuan model untuk mengidentifikasi sebagian besar kasus dalam kelas ini. Namun, untuk kelas dengan penghasilan yang lebih tinggi, terdapat variasi yang lebih signifikan di antara model-model tersebut, dengan XGBoost kembali menonjol karena kemampuannya yang lebih baik dalam mengklasifikasikan kategori ini dibandingkan dengan model lainnya.

Pemilihan model terbaik tentunya tidak hanya bergantung pada akurasi saja, namun juga pada keseluruhan keseimbangan antara precision dan recall, yang diwakili oleh nilai F1-score. Perhatian khusus harus diberikan pada kelas yang lebih sulit diprediksi, yaitu penghasilan >50K, di mana model harus ditingkatkan baik melalui tuning hyperparameter, penggunaan teknik resampling untuk mengatasi ketidakseimbangan kelas, atau eksplorasi fitur yang lebih mendalam. Berikut ini adalah ringkasan dalam bentuk tabel untuk memudahkan perbandingan antar model:
| Model | Akurasi | Precision(<=50K) | Precision(>50K) | Recall(<=50K) | Recall(>50K) | F1-score(<=50K) | F1-score(>50K) |
| ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| Logistic Regression | 84.62% | 0.88 | 0.72 | 0.93 | 0.58 | 0.90 | 0.64 |
| Decision Tree | 82.30% | 0.88 | 0.63 | 0.89 | 0.60 | 0.89 | 0.62 |
| Random Forest | 84.65% | 0.89 | 0.70 | 0.92 | 0.62 | 0.90 | 0.66 |
| XGBoost | 87.06% | 0.89 | 0.77 | 0.94 | 0.64 | 0.92 | 0.70 |
| SVM | 85.23% | 0.87 | 0.75 | 0.94 | 0.56 | 0.91 | 0.64 |

Metrik evaluasi utama yang digunakan untuk mengukur kinerja model dalam proyek ini adalah akurasi dan laporan klasifikasi yang mencakup precision, recall, f1-score, dan support.


**Rubrik/Kriteria Tambahan:**
Berikut penjelasan metrik tersebut:
***Akurasi***
Akurasi adalah metrik yang paling intuitif dan paling umum digunakan untuk mengukur kinerja model. Ini dihitung dengan membagi jumlah prediksi yang benar dengan jumlah total prediksi.
![equation](https://latex.codecogs.com/svg.image?%5Ctext%7BAkurasi%7D=%5Cfrac%7B%5Ctext%7BJumlah%20Prediksi%20Benar%7D%7D%7B%5Ctext%7BJumlah%20Total%20Prediksi%7D)

***Precision***
Precision adalah rasio prediksi positif benar terhadap total prediksi positif. Metrik ini menunjukkan seberapa akurat model dalam memprediksi positif. Precision tinggi menunjukkan bahwa model memiliki sedikit false positive.
![equation](https://latex.codecogs.com/svg.image?%5Ctext%7BPrecision%7D=%5Cfrac%7B%5Ctext%7BTrue%20Positives%7D%7D%7B%5Ctext%7BTrue%20Positives+False%20Positives%7D)

***Recall (Sensitivity)***
Recall mengukur seberapa baik model mengidentifikasi semua positif aktual dari data. Recall tinggi menunjukkan bahwa model berhasil mengidentifikasi sebagian besar positif aktual.
![equation](https://latex.codecogs.com/svg.image?%5Ctext%7BRecall%7D=%5Cfrac%7B%5Ctext%7BTrue%20Positives%7D%7D%7B%5Ctext%7BTrue%20Positives+False%20Negatives)

***F1-Score***
F1-Score adalah rata-rata harmonik dari precision dan recall, menawarkan balance antara keduanya dengan mengambil kedua false positives dan false negatives ke dalam akun. F1-score tinggi menunjukkan bahwa model memiliki kinerja baik dalam hal precision dan recall.
![equation](https://latex.codecogs.com/svg.image?%5Ctext%7BF1-Score%7D=2%5Ctimes%5Cfrac%7B%5Ctext%7BPrecision%7D%5Ctimes%5Ctext%7BRecall%7D%7D%7B%5Ctext%7BPrecision+Recall%7D%7D)

***Support***
Support adalah jumlah actual occurrences dari kelas di dataset yang diberikan. Untuk setiap kelas, menunjukkan berapa banyak sampel pada kelas tersebut yang benar dalam dataset.

   [1]: <https://ieeexplore.ieee.org/abstract/document/8748528>
   [2]: <https://proceedings.neurips.cc/paper_files/paper/2021/file/32e54441e6382a7fbacbbbaf3c450059-Paper.pdf>
   [3]: <https://ieeexplore.ieee.org/abstract/document/10425394>
   [Adult/Census Income Dataset]: <https://archive.ics.uci.edu/dataset/2/adult>
