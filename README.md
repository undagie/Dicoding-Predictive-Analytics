# Laporan Proyek Machine Learning - Muhammad Edya Rosadi
## Domain Proyek
Pasar kerja merupakan sistem dinamis yang dipengaruhi oleh berbagai faktor ekonomi, sosial, dan teknologi, yang semuanya berkontribusi pada pendapatan individu. Faktor-faktor ini termasuk tingkat pendidikan, pengalaman kerja, industri, dan lokasi geografis, serta perubahan dalam teknologi dan permintaan pasar. Menggali data dan tren terkait dapat membantu mengidentifikasi pola dan hubungan yang berpengaruh terhadap kemampuan individu untuk menghasilkan pendapatan[[1]]. Dengan demikian, memahami berbagai aspek ini memungkinkan pembuat kebijakan, perusahaan, dan individu untuk membuat keputusan yang lebih informatif terkait dengan pengembangan karir, peluang pekerjaan, dan strategi ekonomi secara keseluruhan[[2]].

Dalam upaya memahami kompleksitas tersebut, proyek ini berfokus pada analisis dataset Adult/Census yang dihimpun dari UCI Machine Learning Repository. Dataset ini, yang mencakup data sensus dari tahun 1994, berisi informasi tentang atribut demografis dan pekerjaan individu dengan tujuan utama memprediksi apakah pendapatan individu melebihi $50.000 per tahun. Analisis ini penting karena dapat membantu pembuat kebijakan dan perusahaan dalam membuat keputusan yang berinformasi serta membantu individu dalam merencanakan karir mereka[[3]]. Tantangan utama dalam analisis ini termasuk penanganan data kategorikal dan numerik, serta penanganan nilai yang hilang, yang membutuhkan pendekatan analitis yang cermat dan penggunaan teknik pra-pemrosesan yang efektif[[4]].

Mengingat konteks dan tantangan yang ada, proyek ini mengadopsi berbagai algoritma machine learning termasuk Logistic Regression, Decision Trees, Random Forests, SVM, dan XGBoost. Pendekatan ini memungkinkan untuk evaluasi komprehensif terhadap performa model dalam konteks prediksi pendapatan. Penggunaan teknik evaluasi model seperti akurasi, precision, recall, dan F1-score, bersama dengan tuning hyperparameter menggunakan teknik GridSearchCV, merupakan langkah penting dalam memastikan bahwa model yang dikembangkan memiliki kinerja optimal. Dengan demikian, proyek ini tidak hanya bertujuan untuk mengembangkan model prediktif yang akurat tetapi juga memberikan insight tentang faktor-faktor yang mempengaruhi pendapatan, yang dapat bermanfaat bagi stakeholder terkait.

## Business Understanding

**Problem Statements**
Dalam konteks proyek ini, masalah yang ingin diatasi dapat dirumuskan dalam problem statements sebagai berikut:
1. Bagaimana mengidentifikasi faktor-faktor yang paling berpengaruh terhadap penghasilan individu melebihi $50K per tahun?
2. Bagaimana cara mengatasi tantangan ketidakseimbangan data dan missing values yang dapat mempengaruhi kinerja model prediktif?
3. Bagaimana meningkatkan akurasi prediksi penghasilan individu dengan menggunakan pendekatan machine learning?

**Goals**
Tujuan dari proyek ini dirancang untuk menjawab problem statements di atas dengan cara sebagai berikut:
1. Mengidentifikasi dan menganalisis faktor-faktor yang mempengaruhi penghasilan individu menggunakan teknik machine learning.
2. Mengembangkan model prediktif yang sesuai terhadap ketidakseimbangan data dan mampu menangani missing values secara efektif.
3. Meningkatkan akurasi prediksi penghasilan menggunakan pendekatan machine learning.

**Solution Statement**
Untuk mencapai tujuan tersebut, beberapa solusi yang diusulkan meliputi:
1. Penggunaan teknik exploratory data analysis (EDA) untuk mengidentifikasi variabel-variabel yang signifikan terhadap penghasilan dan penerapan teknik feature engineering untuk memaksimalkan informasi yang dapat digunakan oleh model.
2. Implementasi teknik preprocessing data seperti imputasi missing values dan mengurangi dimensionalitas data.
3. Penerapan berbagai model pembelajaran mesin dan tuning hyperparameter untuk menemukan model dengan kinerja terbaik, serta evaluasi model menggunakan metrik yang relevan.

**Benefits and Impact**
Hasil proyek ini memiliki potensi manfaat dan dampak dalam bidang ekonomi dan bisnis: Dalam hal teoritis dapat membantu pembuat kebijakan dalam mengidentifikasi kelompok-kelompok penduduk yang mungkin memerlukan bantuan atau program pengembangan kemampuan untuk meningkatkan pendapatan mereka. Dalam praktiknya, model prediktif yang dikembangkan dapat diintegrasikan ke dalam sistem pengambilan keputusan bisnis untuk memprediksi potensi pendapatan karyawan atau calon karyawan, membantu dalam penyesuaian gaji, dan identifikasi hal terkait lainnya.

## Data Understanding
Pada proyek ini digunakan dataset yang berkaitan dengan informasi pendapatan individu untuk klasifikasi apakah pendapatan seseorang melebihi $50K per tahun atau tidak. Dataset berasal dari repositori UCI Machine Learning ([Adult/Census Income Dataset]) dan dapat digunakan sebagai kasus studi untuk berbagai algoritma klasifikasi dalam pembelajaran mesin. Dataset ini mencakup 14 fitur (lihat tabel 1) seperti usia, jenis pekerjaan, pendidikan, status pernikahan, ras, jenis kelamin, jam kerja per minggu, dan negara asal dengan jumlah data sebanyak 48.842.

**Fitur-fitur pada Adult/Census dataset adalah sebagai berikut:**
Tabel 1. Fitur- fitur pada Adult/Census Dataset
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

Beberapa analisis lanjutan terkait fitur-fitur penting, antara lain:
1. age
![Gambar histrogram dan boxplot fitur usia](https://github.com/undagie/Dicoding-Predictive-Analytics/blob/main/img/age.png?raw=true)
Gambar 1. Histogram dan boxplot fitur usia
- Histogram 'age': menunjukkan bahwa sebagian besar responden berada dalam rentang usia produktif, dengan puncak kepadatan di usia sekitar 20 hingga 50 tahun. Kurva distribusi menunjukkan bahwa data miring ke kanan, dengan lebih sedikit responden berusia di atas 60 tahun.
- Boxplot 'age': memberikan representasi visual tentang distribusi median usia, kuartil, dan adanya outlier. Median usia terletak di sekitar umur 40-an, dan outlier terlihat pada usia yang lebih lanjut, menunjukkan adanya responden yang secara signifikan lebih tua daripada rata-rata populasi dalam dataset.
2. education.num
![Gambar histrogram dan boxplot fitur education.num](https://github.com/undagie/Dicoding-Predictive-Analytics/blob/main/img/educationnum.png?raw=true)
Gambar 2. Histogram dan boxplot fitur education.num
- Histogram 'education.num': menunjukkan bahwa mayoritas responden memiliki 9 hingga 12 tahun pendidikan, yang mencerminkan pendidikan tingkat menengah hingga lulusan SMA. Terdapat pula puncak yang lebih kecil pada 16 tahun, yang menunjukkan jumlah yang signifikan dari responden dengan pendidikan tingkat sarjana.
- Boxplot 'education.num': memberikan representasi bahwa median jumlah tahun pendidikan adalah sekitar 10 tahun, dengan beberapa outlier yang memiliki tahun pendidikan yang sangat rendah atau sangat tinggi. Hal ini menunjukkan variasi dalam tingkat pendidikan yang dicapai oleh individu dalam populasi.
3. hours.per.week
![Gambar histrogram dan boxplot fitur hours.per.week](https://github.com/undagie/Dicoding-Predictive-Analytics/blob/main/img/hoursperweek.png?raw=true)
Gambar 3. Histogram dan boxplot fitur hours.per.week
- Histogram 'hours per week': menunjukkan bahwa jam kerja per minggu terkonsentrasi di sekitar 40 jam, yang merupakan standar waktu kerja penuh di banyak negara. Distribusi menunjukkan adanya konsentrasi tinggi individu yang bekerja dengan jumlah jam standar tersebut.
- Boxplot 'hours per week': menunjukkan bahwa median jam kerja per minggu berkisar di angka standar tersebut, dengan whisker yang meregang ke nilai yang lebih rendah dan lebih tinggi, menandakan bahwa ada variasi dalam jam kerja. Outlier menunjukkan bahwa ada sejumlah responden yang bekerja dengan jam sangat rendah atau sangat tinggi setiap minggunya.

Untuk memahami hubungan antara variabel numerik, digunakan heatmap korelasi, yang memberikan visualisasi tentang seberapa kuat hubungan antar fitur dalam dataset:
![Gambar heatmap korelasi antar fitur](https://github.com/undagie/Dicoding-Predictive-Analytics/blob/main/img/korelasiheatmap.png?raw=true)

Gambar 4. Heatmap korelasi antar fitur
- Korelasi antara 'age' dan 'hours.per.week': Korelasi positif lemah (0.07) menunjukkan bahwa semakin tua usia seseorang, cenderung ada peningkatan kecil dalam jam kerja per minggu.
- Korelasi antara 'education.num' dan 'capital.gain': Ada korelasi positif lemah (0.12), yang dapat diinterpretasikan bahwa individu dengan lebih banyak tahun pendidikan cenderung memiliki capital gain yang lebih tinggi, yang mungkin menunjukkan hubungan antara tingkat pendidikan dan peluang finansial.
- Korelasi antara 'education.num' dan 'hours.per.week': Korelasi positif (0.15) menunjukkan bahwa individu dengan pendidikan lebih tinggi cenderung bekerja lebih banyak jam per minggu, yang bisa berkaitan dengan jenis pekerjaan yang mereka lakukan atau tanggung jawab pekerjaan yang lebih besar.
- 'fnlwgt': Fitur ini memiliki korelasi rendah dengan variabel lain. Nilai korelasinya mendekati nol dengan sebagian besar fitur, yang menunjukkan bahwa tidak ada hubungan linier yang kuat antara 'fnlwgt' dan variabel lain dalam dataset. Karena 'fnlwgt' merupakan estimasi jumlah orang yang diwakili oleh setiap entri dan tidak langsung berhubungan dengan penghasilan individu, fitur ini akan ditinggal saat proses modeling.

## Data Preparation
Proses data preparation merupakan langkah krusial dalam proyek machine learning. Tujuannya adalah untuk mengubah data mentah menjadi format yang lebih cocok untuk pemodelan, sehingga meningkatkan efektivitas dan akurasi dari model yang akan dibangun. Berikut data preparation yang digunakan dalam proyek ini:
1. Penanganan Missing Value
Missing Value adalah kondisi dimana data memiliki beberapa field yang tidak terisi atau kosong. Kondisi ini bisa disebabkan oleh berbagai hal, seperti kesalahan pada saat pengumpulan data atau karena data memang tidak tersedia. Penanganan missing value sangat penting karena keberadaannya dapat mempengaruhi kinerja model pembelajaran mesin.
Teknik-teknik yang dilakukan pada proyek ini untuk penanganan missing value adalah:
    - Drop pada Fitur 'fnlwgt' dan 'income': Fitur 'fnlwgt' dihapus karena tidak memiliki pengaruh signifikan terhadap target model. Fitur 'income' juga dihapus dari dataset fitur dan hanya digunakan sebagai label target.
    - Imputation pada Fitur 'workclass', 'occupation', dan 'native.country': Fitur 'workclass' dan 'occupation' yang memiliki missing value diisi dengan 'Unknown'. Ini dilakukan untuk mempertahankan sampel data tanpa harus menghapus baris yang memiliki informasi penting lainnya.
    - Fitur 'native.country' diisi dengan modusnya (nilai yang paling sering muncul) untuk fitur tersebut. Penggunaan modus sebagai imputasi membantu menjaga distribusi fitur tetap konsisten.
2. Features Encoding
Features Encoding adalah proses konversi fitur kategorikal menjadi fitur numerik agar dapat diproses oleh algoritma machine learning. Kebanyakan algoritma machine learning tidak dapat menangani data kategorikal secara langsung, sehingga encoding menjadi langkah penting dalam persiapan data.
Teknik-teknik yang dilakukan pada proyek ini untuk features encoding adalah:
    - One-Hot Encoding pada Fitur Kategorikal: Teknik One-Hot Encoding digunakan untuk mengubah fitur kategorikal menjadi serangkaian fitur biner (0 atau 1). Setiap kategori pada fitur asli diwakili oleh sebuah fitur baru dengan nilai 1 jika kategori tersebut hadir dan 0 jika tidak. Teknik ini diterapkan pada semua fitur kategorikal seperti 'workclass', 'education', 'marital.status', 'occupation', dll.
    - Standard Scaling pada Fitur Numerik: Fitur numerik diskalakan menggunakan StandardScaler. Ini dilakukan untuk menormalkan distribusi fitur numerik sehingga memiliki mean nol dan standar deviasi satu. Standard scaling membantu meningkatkan kinerja beberapa model pembelajaran mesin yang sensitif terhadap skala fitur.
    - Mengurangi Dimensionalitas dengan Menggabungkan Kategori pada 'native.country': Untuk fitur 'native.country', negara-negara dengan frekuensi kemunculan di bawah ambang batas tertentu digabungkan menjadi satu kategori 'Other'. Hal ini dilakukan untuk mengurangi dimensionalitas data tanpa kehilangan informasi yang signifikan.
3. Penerapan ColumnTransformer dan Preparasi Akhir Data
ColumnTransformer dalam Scikit-Learn memungkinkan untuk menerapkan transformasi yang berbeda pada kolom dataset secara efisien dalam satu langkah. Hal ini sangat berguna dalam kasus di mana dataset memiliki fitur numerik dan kategorikal yang perlu diproses secara berbeda sebelum pemodelan.
Penerapan pada proyek ini:
    - ColumnTransformer dikonfigurasi untuk menerapkan Standard Scaling pada fitur numerik dan One-Hot Encoding pada fitur kategorikal. Ini memastikan bahwa semua fitur disiapkan secara konsisten sesuai dengan kebutuhan algoritma pembelajaran mesin yang digunakan.
    - Setelah semua fitur dipreproses menggunakan ColumnTransformer, dataset dibagi menjadi set pelatihan dan pengujian. Pembagian ini penting untuk validasi model, di mana set pelatihan digunakan untuk melatih model dan set pengujian digunakan untuk mengevaluasi kinerjanya. Pembagian ini dilakukan dengan fungsi train_test_split dari Scikit-Learn, biasanya dengan proporsi pembagian 80% untuk pelatihan dan 20% untuk pengujian.
    - Penghapusan fitur 'fnlwgt' dan target 'income' dari set fitur: Fitur 'fnlwgt' dihapus karena tidak memberikan informasi yang relevan untuk prediksi penghasilan, sementara kolom 'income' dihapus dari set fitur karena merupakan variabel target yang akan diprediksi. Proses ini membantu memfokuskan model pada variabel-variabel yang benar-benar relevan dan penting untuk prediksi.
    - Label Encoding pada Target 'income': Target 'income', yang awalnya dalam format kategorikal ('<=50K' dan '>50K'), dikonversi menjadi format numerik (0 dan 1) menggunakan teknik Label Encoding. Ini merupakan langkah penting karena sebagian besar algoritma machine learning beroperasi pada data numerik. Proses ini mengubah prediksi penghasilan menjadi masalah klasifikasi biner yang dapat diatasi dengan lebih efektif oleh model pembelajaran mesin.

## Modeling
Di bawah ini adalah model-model yang digunakan dan tahapannya:
- Logistic Regression: Menggunakan parameter max_iter=1000 untuk menentukan jumlah iterasi maksimum pada proses pelatihan.
- Decision Tree Classifier: Model ini dijalankan dengan parameter default, membiarkan model menyesuaikan kedalaman pohon dan parameter lainnya berdasarkan data.
- Random Forest Classifier: Sama seperti Decision Tree, namun menggabungkan prediksi dari banyak pohon keputusan untuk meningkatkan akurasi dan mengurangi overfitting.
- XGBoost: Menggunakan XGBClassifier dengan use_label_encoder=False untuk menghindari peringatan deprecation, dan eval_metric='logloss' untuk mengevaluasi kinerja model selama pelatihan.
- SVM (Support Vector Machine): Digunakan untuk klasifikasi dengan SVC(probability=True) memungkinkan estimasi probabilitas, yang memerlukan pelatihan internal tambahan untuk memperkirakan probabilitas.

Berikut tabel kelebihan dan kekurangan setiap algoritma yang digunakan:
Tabel 2. Kelebihan dan kekurangan algoritma yang diusulkan
| Algoritma | Kelebihan | Kekurangan |
| ------ | ------ | ------ |
| Logistic Regression | - Sederhana dan mudah untuk diimplementasikan.<br> - Cepat dalam pelatihan dan prediksi.<br>- Memiliki interpretasi yang baik. |- Kurang efektif pada ruang fitur yang sangat besar atau dataset yang sangat kompleks.<br>- Rentan terhadap overfitting pada dataset dengan fitur yang sangat banyak. |
| Decision Tree |- Mudah untuk diinterpretasikan dan dijelaskan.<br>- Dapat menangani data kategorikal dan numerik.<br>- Tidak memerlukan normalisasi data. | - Rentan terhadap overfitting, terutama pada pohon yang sangat dalam.<br>- Varians yang tinggi dapat menyebabkan perubahan besar pada struktur pohon dengan perubahan kecil pada data. |
| Random Forest | - Mengurangi overfitting melalui ensemble pohon.<br>- Fleksibel dan dapat digunakan untuk klasifikasi dan regresi.<br>- Performa yang baik pada banyak masalah. | - Lebih lambat dalam pelatihan dan prediksi dibandingkan dengan model yang lebih sederhana.<br>- Lebih sulit untuk diinterpretasikan. |
| XGBoost | - Optimisasi untuk komputasi yang cepat dan penggunaan memori yang efisien.<br>- Mendukung regularisasi untuk mengurangi overfitting.<br>- Fleksibel dan dapat menyesuaikan banyak masalah data. | - Meskipun relatif cepat, bisa menjadi sumber daya yang intensif dan memakan waktu pada dataset sangat besar.<br>- Memiliki kurva belajar yang lebih curam karena banyaknya parameter yang dapat dituning. |
| Support Vector Machine (SVM) | - Efektif dalam ruang dimensi tinggi.<br>- Efektif pada kasus di mana jumlah dimensi lebih besar dari jumlah sampel.<br>- Memiliki keflexibelan dalam pemilihan fungsi kernel. | - Memerlukan pemilihan kernel yang tepat.<br>- Rentan terhadap overfitting pada fitur noise yang banyak.<br>- Waktu dan sumber daya komputasi yang tinggi untuk dataset besar. |

Setelah evaluasi hasil dan mempertimbangkan kelebihan dan kekurangan pada tabel 2 di atas, maka dilakukan tuning hyperparameter pada model terpilih (XGBoost, Random Forest, dan SVM) menggunakan GridSearchCV. Proses ini mencari kombinasi parameter terbaik yang menghasilkan akurasi tertinggi.
- XGBoost: Grid search dilakukan pada n_estimators, max_depth, dan learning_rate.
- Random Forest: Grid search meliputi n_estimators, max_depth, dan min_samples_split.
- SVM: Menyesuaikan parameter C, gamma, dan kernel.

## Evaluation
Metrik evaluasi utama yang digunakan untuk mengukur kinerja model dalam proyek ini adalah akurasi dan laporan klasifikasi yang mencakup precision, recall, f1-score, dan support. Berikut penjelasan dan perhitungan metrik tersebut:

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

Berdasarkan analisis terhadap laporan klasifikasi dari berbagai model machine learning, dapat ditarik beberapa kesimpulan mengenai performa masing-masing model dalam tugas klasifikasi penghasilan menggunakan dataset Adult/Census Dataset. Model XGBoost menunjukkan performa terbaik dengan akurasi tertinggi sebesar 87.06%, diikuti oleh Random Forest dengan akurasi 84.65%, SVM dengan 85.23%, dan Decision Tree dengan 82.30%. XGBoost unggul bukan hanya dalam hal akurasi tetapi juga dalam aspek precision dan F1-score, khususnya untuk kelas penghasilan yang lebih tinggi (>50K).

Dalam aspek precision dan recall untuk kelas dengan penghasilan lebih rendah (<=50K), semua model cenderung menampilkan hasil yang kuat, dengan nilai recall yang sangat tinggi yang menunjukkan kemampuan model untuk mengidentifikasi sebagian besar kasus dalam kelas ini. Namun, untuk kelas dengan penghasilan yang lebih tinggi, terdapat variasi yang lebih signifikan di antara model-model tersebut, dengan XGBoost kembali menonjol karena kemampuannya yang lebih baik dalam mengklasifikasikan kategori ini dibandingkan dengan model lainnya.

Pemilihan model terbaik tentunya tidak hanya bergantung pada akurasi saja, namun juga pada keseluruhan keseimbangan antara precision dan recall, yang diwakili oleh nilai F1-score. Perhatian khusus harus diberikan pada kelas yang lebih sulit diprediksi, yaitu penghasilan >50K, di mana model harus ditingkatkan baik melalui tuning hyperparameter, penggunaan teknik resampling untuk mengatasi ketidakseimbangan kelas, atau eksplorasi fitur yang lebih mendalam. Berikut ini adalah ringkasan dalam bentuk tabel untuk memudahkan perbandingan antar algoritma:
Tabel 3. Perbandingan hasil antar algoritma yang diusulkan
| Algoritma | Akurasi | Precision(<=50K) | Precision(>50K) | Recall(<=50K) | Recall(>50K) | F1-score(<=50K) | F1-score(>50K) |
| ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| Logistic Regression | 84.62% | 0.88 | 0.72 | 0.93 | 0.58 | 0.90 | 0.64 |
| Decision Tree | 82.30% | 0.88 | 0.63 | 0.89 | 0.60 | 0.89 | 0.62 |
| Random Forest | 84.65% | 0.89 | 0.70 | 0.92 | 0.62 | 0.90 | 0.66 |
| XGBoost | 87.06% | 0.89 | 0.77 | 0.94 | 0.64 | 0.92 | 0.70 |
| SVM | 85.23% | 0.87 | 0.75 | 0.94 | 0.56 | 0.91 | 0.64 |

Evaluasi selanjutnya adalah hasil dari tuning hyperparameter untuk 3 model dengan akurasi terbaik (Random Forest, XGBoost, dan SVM):
- Untuk XGBoost, kombinasi hyperparameter terbaik yang ditemukan adalah learning_rate sebesar 0.1, max_depth sebesar 7, dan n_estimators sebesar 200. Dengan set parameter ini, akurasi meningkat menjadi 87.285%, yang menegaskan posisinya sebagai model dengan performa terbaik di antara yang lain.
- Random Forest juga menunjukkan peningkatan dalam akurasi setelah tuning, dengan akurasi terbaik yang diperoleh sebesar 86.390%. Hyperparameter yang menghasilkan performa terbaik adalah max_depth sebesar 20, min_samples_split sebesar 10, dan n_estimators sebesar 100.
- Optimasi pada SVM menghasilkan akurasi sebesar 85.634% dengan hyperparameter terbaik: C sebesar 1, gamma dengan nilai 'scale', dan menggunakan kernel 'rbf'.

Proyek ini berfokus pada identifikasi faktor-faktor yang mempengaruhi penghasilan individu dengan tujuan untuk memprediksi apakah penghasilan individu melebihi $50K per tahun. Problem statement mencakup tantangan seperti ketidakseimbangan kelas, kebutuhan preprocessing yang kompleks, dan seleksi fitur yang tepat untuk memaksimalkan kinerja model. Tujuan utama dari proyek ini adalah untuk mengembangkan model yang mampu secara akurat mengklasifikasikan individu berdasarkan tingkat penghasilan mereka, dengan harapan model tersebut dapat digunakan sebagai alat bantu dalam pembuatan kebijakan dan strategi intervensi ekonomi.

Hasil eksperimen menunjukkan bahwa model XGBoost berhasil mencapai akurasi tertinggi (87.285% setelah tuning hyperparameter), yang menunjukkan bahwa proyek ini berhasil mengembangkan model dengan kemampuan prediksi yang kuat. Tingginya nilai precision dan F1-score, khususnya untuk kelas penghasilan yang lebih tinggi, menegaskan bahwa model ini efektif dalam mengidentifikasi individu dengan penghasilan >$50K, yang merupakan salah satu kriteria utama keberhasilan proyek ini.

Dari hasil eksperimen, dapat disimpulkan bahwa proyek ini berhasil dalam mencapai tujuan utamanya. Model XGBoost, dengan tuning hyperparameter yang tepat, menawarkan keseimbangan yang baik antara akurasi dan kemampuan untuk mengklasifikasikan kedua kelas penghasilan dengan efektif. Namun, masih terdapat ruang untuk peningkatan, terutama dalam aspek recall untuk kelas penghasilan >$50K. Kedepannya, dapat dilakukan eksplorasi lebih lanjut mengenai teknik resampling untuk mengatasi ketidakseimbangan kelas dan eksplorasi fitur yang lebih mendalam untuk meningkatkan kinerja model.

## Daftar Referensi
[[1]] N. Chakrabarty dan S. Biswas, "Statistical Approach to Adult Census Income Level Prediction," in Proceedings of the International Conference on Machine Learning and Data Engineering (iCMLDE), IEEE, 2018.<br>
[[2]] M. A. Islam, A. Nag, N. Roy, A. R. Dey, S. M. F. A. Fahim, dan A. Ghosh, "An Investigation into the Prediction of Annual Income Levels Through the Utilization of Demographic Features Employing the Modified UCI Adult Dataset," in Proceedings of the International Conference on Advances in Electrical Engineering (ICAEE), IEEE, 2020.<br>
[[3]] F. Ding, M. Hardt, J. Miller, dan L. Schmidt, "Retiring Adult: New Datasets for Fair Machine Learning," in Advances in Neural Information Processing Systems (NeurIPS), 2021.<br>
[[4]] S. R. G. Shashidhar, "Big data in healthcare: management, analysis and future prospects," in Journal of Big Data, vol. 8, no. 1, Springer, 2021.<br>

   [1]: <https://ieeexplore.ieee.org/abstract/document/8748528>
   [2]: <https://ieeexplore.ieee.org/abstract/document/10425394>
   [3]: <https://proceedings.neurips.cc/paper_files/paper/2021/file/32e54441e6382a7fbacbbbaf3c450059-Paper.pdf>
   [4]: <https://link.springer.com/article/10.1186/s40537-021-00516-9>
   [Adult/Census Income Dataset]: <https://archive.ics.uci.edu/dataset/2/adult>
