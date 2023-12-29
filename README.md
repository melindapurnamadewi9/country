# Laporan Proyek Machine Learning
### Nama  : Melinda Purnama Dewi
### Nim   : 211351082
### Kelas : Pagi B

## Domain Proyek

Unsupervised Learning on Country Data, projek ini dapat digunakan untuk mengkategorikan negara-negara menggunakan faktor sosial-ekonomi dan kesehatan yang menentukan pembangunan negara secara keseluruhan.

## Business Understanding

Dapat berkomitmen untuk memerangi kemiskinan dan menyediakan fasilitas dan bantuan dasar bagi masyarakat di negara-negara terbelakang pada saat terjadi bencana dan bencana alam.


### Problem Statements

Memungkinkan  CEO LSM tersebut perlu memutuskan bagaimana menggunakan dana tersebut secara strategis dan efektif. 


### Goals

- Anda  dapat  mengkategorikan negara menggunakan beberapa faktor sosial-ekonomi dan kesehatan yang menentukan perkembangan negara secara keseluruhan. 
- Kemudian CEO harus mengambil keputusan untuk memilih negara yang paling membutuhkan bantuan dan anda  menyarankan negara-negara mana yang paling perlu menjadi fokus CEO.


## Data Understanding

[Unsupervised Learning on Country Data] https://www.kaggle.com/datasets/rohan0301/unsupervised-learning-on-country-data/data
 

### Variabel-variabel pada Heart Failure Prediction Dataset adalah sebagai berikut:
- Country       : Nama negara (float)
- Child_mort    : Kematian anak dibawah usia 5 tahun per 1000 kelahiran hidup (float)
- Exports       : Ekspor barang dan jasa per kapita. Diberikan sebagai %usia PDB per kapita (float)
- Health        : Total pengeluaran kesehatan per kapita,diberikan sebagai %usia PDB per kapita (float)
- Imports       : Impor barang dan jasa per kapita. Diberikan sebagai %usia PDB per kapita (float)
- Income        : Pendapatan bersih per orang (int)
- Inflation     : Pengukuran tingkat pertumbuhan tahunan Total PDB (float)
- Life_expec    : Rata-rata lama hidup seorang anak yang baru lahir jika pola kematian saat ini ingin tetap sama (float)
- Total_fer     : Jumlah anak yang akan dilahirkan oleh setiap wanita jika angka kesuburan usia saat ini tetap sama (float)
- Gdpp          : PDB per kapita. Dihitung sebagai Total PDB dibagi dengan total populasi.(int)


## Data Preparation
## Data Collection

Untuk data colletion ini, saya mendapatkan dataset yang nanti dapat digunakan dari website kaggle dengan nama DUnsupervised Learning on Country Data, jika anda tertarik dengan dataset tesebut bisa klik diatas.

## Data Discovery and Profiling

Untuk bagian ini saya menggunakan Teknik EDA

Tentukan library yang digunakan, disini saya menggunakan google collab .
## Import Library
```bash
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
```
Karena kita menggunakan google collab untuk itu kita mengerjakannya, maka kita akan import files juga 
```bash
from google.colab import files
```
Lalu mengupload token kaggle agar nanti kita dapat  mendownload sebuah datasets dari kaggle melalui google collab
``bash
file.upload()
```
Setelah mengupload filenya, maka kita selanjutnya dengan membuat sebuah folder untuk menyimpan file kaggle.json yang sudah diupload tadi
```bash
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!ls ~/.kaggle
```
Lalu mari kita download datasets nya
```bash
!kaggle datasets download -d rohan0301/unsupervised-learning-on-country-data --force
```
Selanjutnya kita harus extract file yang tadi telah didownload
```bash
!mkdir rohan0301
!unzip unsupervised-learning-on-country-data.zip
```
# Data Discovery

Lanjut dengan memasukkan file csv yang telah diextract pada sebuah variable
```bash
df = pd.read_csv('Country-data.csv')
```
Memanggil data yang sudah diubah 
```bash
df.head()
```
```bash
df['country'].value_counts().sum()
```
```bash
df = df.drop(['country'], axis=1)
```
```bash
df.to_csv('Country.csv', index=False)
```
Untuk melihat mengenai type data dari masing masing kolom kita bisa menggunakan property info
```bash
df.info()
```
```bash
X = df
```
```bash
X
```
```bash
df['gdpp'].sum()
````
Selanjutnya kita masukan eda (minimal 5)
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
![image](https://github.com/melindapurnamadewi9/country/assets/148632928/23ab887d-0691-4600-8f5c-66d92d5c32ff)

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
![image](https://github.com/melindapurnamadewi9/country/assets/148632928/619ef37a-786a-4c1d-8862-65168b5fd58f)

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
![image](https://github.com/melindapurnamadewi9/country/assets/148632928/d51ebfec-6747-453b-8daa-d92747a5bfee)

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
![image](https://github.com/melindapurnamadewi9/country/assets/148632928/b3e11e85-5f60-4dae-bd4f-0824c1d54067)

```bash
plt.figure(figsize=(10, 6))
sns.scatterplot(data=X, x=X['health'], y=X['life_expec'], s=100, color='green', alpha=0.8)
plt.title('Scatter Plot: health dan life_expec')
plt.xlabel('health')
plt.ylabel('life_expec')
plt.show()
```
![image](https://github.com/melindapurnamadewi9/country/assets/148632928/1362172d-1050-4699-8cbb-52a5ae68e125)

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
![image](https://github.com/melindapurnamadewi9/country/assets/148632928/73fc43bc-16be-4db0-aeb2-49d9157d7856)

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
![image](https://github.com/melindapurnamadewi9/country/assets/148632928/eaa03842-d3aa-4831-98ce-8911bac38e4a)

## Modeling

Model k-means adalah algoritma dalam machine learning yang digunakan untuk melakukan klasterisasi atau pengelompokan data.

Tujuan utama dari algoritma ini adalah untuk membagi himpunan data menjadi beberapa kelompok, yang disebut klaster, sehingga objek-objek dalam satu klaster memiliki kesamaan yang tinggi, sedangkan objek-objek antar klaster memiliki kesamaan yang rendah

```bash 
n_clust = 5
kmean = KMeans(n_clusters=n_clust).fit(X)
X['Labels'] = kmean.labels_
```
![image](https://github.com/melindapurnamadewi9/country/assets/148632928/1f555912-94e5-4afa-a0e1-ec95d9c9d981)

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

```bash
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k)
    labels = kmeans.fit_predict(X)
    silhouette_avg = silhouette_score(X, labels)
    print(f"For n_clusters = {k}, the average silhouette_score is : {silhouette_avg}")

```
![image](https://github.com/melindapurnamadewi9/country/assets/148632928/fc5a3f5f-a9f7-4444-b0d1-111a7efea874)


## Evaluation

K-means clustering adalah salah satu algoritma klasterisasi yang populer dalam dunia machine learning dan analisis data. 
Tujuannya adalah membagi himpunan data menjadi beberapa kelompok atau klaster berdasarkan kesamaan antar data. 
Algoritma ini sangat sederhana namun efektif


kmeans = KMeans(n_clusters=k, n_init=10)  # Set the value of n_init explicitly

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



_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.

