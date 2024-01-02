# Laporan Proyek Machine Learning
### Nama  : Melinda Purnama Dewi
### Nim   : 211351082
### Kelas : Pagi B


## Domain Proyek

Unsupervised Learning on Country Data, projek ini dapat digunakan untuk untuk mengkategorikan negara-negara, yang dapat dilihat dari aspek sosial ekonomi dan kesehatan yang dapat menentukan pembangunan sebuah negara secara keseluruhan.


## Business Understanding

Ketidakmungkinan seseorang membantu negara -negara yang memerlukan bantuan bencana, dengan adanya HELP International merupakan LSM kemanusiaan internasional yang berkomitmen untuk memerangi kemiskinan dan menyediakan fasilitas dan bantuan dasar bagi masyarakat di negara-negara terbelakang pada saat terjadi bencana dan bencana alam.


### Problem Statements

- HELP International merupakan sebuah organisasi LSM kemanusiaan internasional yang  telah mampu mengumpulkan sekitar $10 juta.
- CEO LSM tersebut perlu memutuskan bagaimana menggunakan dana tersebut secara strategis dan efektif.


### Goals

- Anda dapat memberi saran kepada CEO untuk mengambil keputusan untuk memilih negara yang paling membutuhkan bantuan dengan mengkategorikan negara menggunakan beberapa faktor sosial-ekonomi dan kesehatan yang menentukan perkembangan negara secara keseluruhan. 
- Anda dapat mengambil keputusan untuk memilih negara yang paling membutuhkan bantuan dan dapat menyarankan negara-negara mana yang paling perlu menjadi fokus CEO.

## Solution statements

- Pengembangan platfrom Unsupervised Learning on Country Data yaitu memberikan landasan untuk memanfaatkan teknik pembelajaran tanpa pengawasan pada data negara, menawarkan wawasan dan memfasilitasi pengambilan keputusan yang tepat di berbagai bidang. Data yang berasal dari kaggle.com yang memberikan pengguna dapat mengakses dengan cepat dan mudah.
- Model yang dihasilkan dari dataset ini menggunakan metode K-means 


## Data Understanding

Dataset yang saya gunakan berasal dari kaggle, yang berisi 167 baris dan 10 kolom.

[Unsupervised Learning on Country Data]

https://www.kaggle.com/datasets/rohan0301/unsupervised-learning-on-country-data/data


### Variabel-variabel pada Heart Failure Prediction Dataset adalah sebagai berikut:

- Country       : Nama negara [Tipe Data:float]
- Child_mort    : Kematian anak dibawah usia 5 tahun per 1000 kelahiran hidup [Tipe Data:float]
- Exports       : Ekspor barang dan jasa per kapita. Diberikan sebagai %usia PDB per kapita [Tipe Data:float]
- Health        : Total pengeluaran kesehatan per kapita,diberikan sebagai %usia PDB per kapita [Tipe Data:float]
- Imports       : Impor barang dan jasa per kapita. Diberikan sebagai %usia PDB per kapita [Tipe Data:float)
- Income        : Pendapatan bersih per orang [Tipe Data:int]
- Inflation     : Pengukuran tingkat pertumbuhan tahunan Total PDB [Tipe Data:float]
- Life_expec    : Rata-rata lama hidup seorang anak yang baru lahir jika pola kematian saat ini ingin tetap sama [Tipe Data:float]
- Total_fer     : Jumlah anak yang akan dilahirkan oleh setiap wanita jika angka kesuburan usia saat ini tetap sama [Tipe Data:float]
- Gdpp          : PDB per kapita. Dihitung sebagai Total PDB dibagi dengan total populasi.[Tipe Data:int]


## Data Preparation
## Data Collection

Untuk data colletion ini, saya mendapatkan dataset ini dari website kaggle dengan nama Unsupervised Learning on Country Data, jika anda tertarik dengan dataset tesebut bisa klik diatas.

## Data Discovery and Profiling

Untuk bagian ini saya menggunakan Teknik EDA

Tentukan library yang digunakan, disini saya menggunakan google collab.

Import Library

```bash
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

```
Karena kita menggunakan google collab untuk itu kita menjalankanya, maka kita akan import files.

```bash
from google.colab import files.
```
Lalu mengupload token kaggle agar nanti kita dapat setelah mendownload sebuah datasets dari kaggle melalui google collab.

```bash
file.upload()
```
Setelah mengupload filenya, maka akan lanjut dengan membuat sebuah folder untuk menyimpan file kaggle.json yang sudah diupload tadi.

```bash
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!ls ~/.kaggle
```
Lalu mari kita download datasets nya.

```bash
!kaggle datasets download -d rohan0301/unsupervised-learning-on-country-data --force
```
Selanjutnya kita harus extract file yang tadi telah didownload.

```bash
!mkdir rohan0301
!unzip unsupervised-learning-on-country-data.zip
```

Lanjut dengan memasukkan file csv yang telah diextract pada sebuah variable.

```bash
df = pd.read_csv('Country-data.csv')
```

Untuk melihat mengenai type data dari masing masing kolom kita bisa menggunakan property info.

```bash
df.info()
```
Selanjutnya kita masukan EDA (Minimal 5)

```bash
plt.figure(figsize=(10, 6))
sns.histplot(data=X, x=X['exports'], kde=True, color='blue', label='Export')
sns.histplot(data=X, x=X['imports'], kde=True, color='orange', label='Import')
plt.title('Export dan import')
plt.xlabel('Nilai')
plt.ylabel('Frekuensi')
plt.legend()
plt.show()
```
![image](https://github.com/melindapurnamadewi9/country/assets/148632928/7ccd7da6-659a-481b-b1e4-a98b521df129)

```bash
plt.figure(figsize=(10, 6))
plt.plot(X['exports'], label='exports', marker='o', linestyle='-')
plt.plot(X['inflation'], label='inflation', marker='o', linestyle='-')
plt.title('exports dan inflation')
plt.xlabel('Data Point')
plt.ylabel('Nilai')
plt.legend()
plt.show()
```
![image](https://github.com/melindapurnamadewi9/country/assets/148632928/495fd392-6a5f-43b3-add1-1b4bcb100eb8)

```bash
plt.figure(figsize=(10, 6))
plt.plot(X['imports'], label='imports', marker='o', linestyle='-')
plt.plot(X['inflation'], label='inflation', marker='o', linestyle='-')
plt.title('imports dan inflation')
plt.xlabel('Data Point')
plt.ylabel('Nilai')
plt.legend()
plt.show()
```
![image](https://github.com/melindapurnamadewi9/country/assets/148632928/00ecb84b-118e-4a6d-9af2-ef6bf1213e3d)

```bash
plt.figure(figsize=(10, 6))
plt.plot(X['income'], label='income', marker='o', linestyle='-')
plt.plot(X['gdpp'], label='gdpp', marker='o', linestyle='-')
plt.title('income dan gdpp')
plt.xlabel('Data Point')
plt.ylabel('Nilai')
plt.legend()
plt.show()
```
![image](https://github.com/melindapurnamadewi9/country/assets/148632928/f4f0dda7-9a2b-4bdb-8012-040731f44348)

```bash
plt.figure(figsize=(10, 6))
sns.scatterplot(data=X, x=X['health'], y=X['life_expec'], s=100, color='green', alpha=0.8)
plt.title('Scatter Plot: health dan life_expec')
plt.xlabel('health')
plt.ylabel('life_expec')
plt.show()
```
![image](https://github.com/melindapurnamadewi9/country/assets/148632928/d00b46ac-f5ba-49ed-97ee-906e8a990a9b)

```bash

labels = ['exports', 'imports']
quantity = [6965, 7830]
colors = ['yellowgreen', 'gold']

plt.title('Export dan import')
plt.pie(quantity, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=90)

plt.axis('equal')
plt.show()
```
![image](https://github.com/melindapurnamadewi9/country/assets/148632928/64e3a075-a5da-4e4b-90e3-e94c61b864ed)

```bash
from matplotlib import style

style.use('ggplot')

x = [0, 1]
y = [2863163, 2165014]

fig, ax = plt.subplots()

ax.bar(x, y, align='center')

ax.set_title('Income dan gdpp')
ax.set_ylabel('income')
ax.set_xlabel('gdpp')

ax.set_xticks(x)
ax.set_xticklabels(("income", "gdpp"))

plt.show()
```
![image](https://github.com/melindapurnamadewi9/country/assets/148632928/7569f09a-5517-4dad-b0c7-705dfdc8f7e6)


## Modeling

K-means merupakan salah satu algoritma yang bersifat unsupervised learning. K-Means memiliki fungsi untuk mengelompokkan data kedalam data cluster. 

K-Means Clustering adalah suatu metode penganalisaan data atau metode Data Mining yang melakukan proses pemodelan unssupervised learning dan menggunakan metode yang mengelompokan data berbagai partisi. K Means Clustering memiliki objective yaitu meminimalisasi object function yang telah di atur pada proses clasterisasi. Dengan cara minimalisasi variasi antar 1 cluster dengan maksimalisasi variasi dengan data di cluster lainnya.

Rumus 

![jarak](https://github.com/melindapurnamadewi9/country/assets/148632928/3a19d268-95e1-495f-bedc-dc3fc59d8abf)


```bash 
n_clust = 5
kmean = KMeans(n_clusters=n_clust).fit(X)
X['Labels'] = kmean.labels_
```

```bash
plt.figure(figsize=(10, 8))
scatterplot = sns.scatterplot(x='income', y='gdpp', hue='Labels', size='Labels', data=X, palette='hls', markers=True)
scatterplot.legend(title='Clusters', loc='upper right', bbox_to_anchor=(1.2, 1))

for label in X['Labels'].unique():
    cluster_mean = X[X['Labels'] == label][['income', 'gdpp']].mean()
    plt.annotate(label,
                 cluster_mean,
                 ha='center',
                 va='center',
                 color='black',
                 size=10,
                 weight='bold',
                 backgroundcolor='white',
                 bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))

plt.title('Cluster Analysis')
plt.xlabel('Income')
plt.ylabel('Score')

plt.show()
```
![image](https://github.com/melindapurnamadewi9/country/assets/148632928/cdb6f2ba-b008-4aa0-b838-6865e7262faa)

```bash
silhouette_scores = []

for num_clusters in range(2, 10):
    kmeans = KMeans(n_clusters=num_clusters)
    labels = kmeans.fit_predict(X)
    silhouette_avg = silhouette_score(X, labels)
    silhouette_scores.append(silhouette_avg)

plt.figure(figsize=(8, 5))
plt.plot(range(2, 10), silhouette_scores, marker='o')
plt.title('Silhouette Scores for Different Numbers of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.show()
```
![image](https://github.com/melindapurnamadewi9/country/assets/148632928/7d4bd670-2017-484c-8c21-86fbe34f67aa)

```bash
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k)
    labels = kmeans.fit_predict(X)
    silhouette_avg = silhouette_score(X, labels)
    print(f"For n_clusters = {k}, the average silhouette_score is : {silhouette_avg}")

```

## Evaluation

Didalam Visualisasi hasil algoritma terdapat 8 cluster, dari cluster yang terkecil dengan score siluet 0.33 dan yang terbesar score siluet 0.79. 
Cluster yang optimal ada di cluster 3 dengan score siluet 0.79

```bash
kmeans = KMeans(n_clusters=k, n_init=10)  # Set the value of n_init explicitly
```

```bash

from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples, silhouette_score
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=42)

# Loop melalui berbagai nilai n_clusters
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    silhouette_avg = silhouette_score(X, labels)
    sample_silhouette_values = silhouette_samples(X, labels)

    plt.figure(figsize=(6, 4))
    plt.title(f'KMeans Clustering dengan {k} Klaster\nSkor Siluet: {silhouette_avg:.2f}')

    y_lower = 10
    for i in range(k):
        ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = plt.cm.nipy_spectral(float(i) / k)
        plt.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)

        plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    plt.xlabel("Nilai Koefisien Siluet")
    plt.ylabel("Label Klaster")
    plt.axvline(x=silhouette_avg, color="red", linestyle="--")
    plt.yticks([])
    plt.show()
```

![image](https://github.com/melindapurnamadewi9/country/assets/148632928/0d465acd-af58-405b-8ff5-0d065aa7cb9b)
![image](https://github.com/melindapurnamadewi9/country/assets/148632928/9799bb82-a743-4210-b095-c539dcca525d)
![image](https://github.com/melindapurnamadewi9/country/assets/148632928/4d97a180-6efb-46fd-aca0-07db375913e4)
![image](https://github.com/melindapurnamadewi9/country/assets/148632928/e1c41efe-c416-4c3d-8e82-91794fd7febf)
![image](https://github.com/melindapurnamadewi9/country/assets/148632928/e69e4782-8d2a-4f72-9d6c-9e4c1928051e)
![image](https://github.com/melindapurnamadewi9/country/assets/148632928/c04002b7-5b0f-456b-aded-cb8b2de15d08)
![image](https://github.com/melindapurnamadewi9/country/assets/148632928/26171735-7a13-4ebb-acc7-9c86c33a0026)
![image](https://github.com/melindapurnamadewi9/country/assets/148632928/edf35619-2bc2-4fef-b3ca-cedd68ba485e)

## Deployment
Pada bagian ini anda memberikan link project yang diupload melalui streamlit share. boleh ditambahkan screen shoot halaman webnya.

Github   : https://github.com/melindapurnamadewi9/country/tree/main

Stremlit : https://country-uzq4p2pkgmlvodd9564wzm.streamlit.app/

![Screenshot (218)](https://github.com/melindapurnamadewi9/country/assets/148632928/8b87b2b2-af28-4a76-97a4-eac8047c98f5)
![Screenshot (219)](https://github.com/melindapurnamadewi9/country/assets/148632928/07da377a-bec4-432f-80fc-0e74c616d602)





















