import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score, silhouette_samples
import pandas as pd
import numpy as np

# Fungsi untuk menjalankan KMeans clustering dan menampilkan hasil
def run_kmeans(n_clusters, X):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)
    silhouette_avg = silhouette_score(X, labels)
    
    # Menampilkan dataset asli dengan label cluster yang ditetapkan
    original_data = pd.DataFrame(X, columns=['income', 'gdpp'])
    original_data['Cluster'] = labels
    st.subheader("Dataset Original dengan Label Cluster:")
    st.write(original_data)

    # Menampilkan skor siluet
    st.subheader(f"Skor Siluet: {silhouette_avg:.2f}")

    # Menampilkan elbow plot
    inertias = []
    for k in range(1, n_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, n_clusters + 1), inertias, marker='o')
    ax.set_title('Elbow Plot untuk KMeans Clustering')
    ax.set_xlabel('Jumlah Cluster')
    ax.set_ylabel('Inertia')

    st.pyplot(fig)

# Aplikasi Streamlit
def main():
    st.title("Hasil KMeans Clustering")
    
    

    # Menghasilkan dataset contoh
    data = {
        'income': [1610, 9930, 12900, 5900, 19100, 18700, 6700, 41400, 43200, 16000],
        'gdpp': [553, 4090, 4460, 3530, 12200, 10300, 3220, 51900, 46900, 5840],
    }
    X = pd.DataFrame(data)
    X1 = pd.read_csv('Country-label.csv')
    # X2 = X1['income','gdpp', 'Labels']
    
    chart_data = X1

    st.scatter_chart(
    chart_data,
    x='income',
    y='gdpp',
    color='Labels' )
    
    st.scatter_chart(chart_data)
    

    # Sidebar
    st.sidebar.header("Pengaturan")
    n_clusters = st.sidebar.slider("Pilih Jumlah Maksimal Cluster", 2, 10, value=4)

    # Menjalankan KMeans dan menampilkan hasil
    run_kmeans(n_clusters, X)
    

if __name__ == "__main__":
    main()
