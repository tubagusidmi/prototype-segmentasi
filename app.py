import streamlit as st
import math
import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# =====================================================
# KONFIGURASI HALAMAN
# =====================================================
st.set_page_config(
    page_title="Prototipe Segmentasi Anak Putus Sekolah",
    layout="wide"
)

st.title("📊 Prototipe Segmentasi Anak Putus Sekolah")
st.markdown(
    "Aplikasi web sederhana untuk melakukan klasterisasi menggunakan algoritma **K-Means**."
)

st.divider()

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.header("⚙️ Pengaturan")
uploaded_file = st.sidebar.file_uploader("Upload Dataset CSV", type=["csv"])
K = st.sidebar.slider("Jumlah Cluster (K)", min_value=2, max_value=8, value=4)
MAX_ITER = 100

# =====================================================
# FUNGSI K-MEANS
# =====================================================
def euclidean(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(len(a))))

def init_centroids(data, k):
    return random.sample(data, k)

def assign_clusters(data, centroids):
    clusters = [[] for _ in centroids]
    labels = []

    for idx, point in enumerate(data):
        distances = [euclidean(point, c) for c in centroids]
        cluster_idx = distances.index(min(distances))
        clusters[cluster_idx].append((idx + 1, point))
        labels.append(cluster_idx)

    return clusters, labels

def compute_centroids(clusters, dim):
    centroids = []
    for cluster in clusters:
        if not cluster:
            centroids.append([0] * dim)
        else:
            centroid = [
                sum(p[1][i] for p in cluster) / len(cluster)
                for i in range(dim)
            ]
            centroids.append(centroid)
    return centroids

# =====================================================
# PROSES UTAMA
# =====================================================
if uploaded_file is not None:

    try:
        df = pd.read_csv(uploaded_file, header=None)
        dataset = df.values.tolist()
    except Exception:
        st.error("❌ Gagal membaca file CSV.")
        st.stop()

    if len(dataset) == 0:
        st.error("❌ Dataset kosong.")
        st.stop()

    st.success(f"📁 Dataset dimuat: {len(dataset)} data, {len(dataset[0])} variabel")

    if st.button("🚀 Proses K-Means"):

        random.seed(42)
        centroids = init_centroids(dataset, K)

        for _ in range(MAX_ITER):
            clusters, labels = assign_clusters(dataset, centroids)
            new_centroids = compute_centroids(clusters, len(dataset[0]))
            if centroids == new_centroids:
                break
            centroids = new_centroids

        # =====================================================
        # VISUALISASI PCA
        # =====================================================
        st.subheader("📈 Visualisasi PCA Scatter Plot")

        pca = PCA(n_components=2)
        data_2d = pca.fit_transform(dataset)

        fig, ax = plt.subplots()
        for i in range(K):
            points = data_2d[[j for j in range(len(labels)) if labels[j] == i]]
            ax.scatter(points[:, 0], points[:, 1], label=f"Cluster {i+1}")

        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.legend()
        st.pyplot(fig)

        # =====================================================
        # TABEL HASIL
        # =====================================================
        st.subheader("📋 Tabel Hasil Klasterisasi")

        hasil = df.copy()
        hasil["Cluster"] = [l + 1 for l in labels]
        st.dataframe(hasil, use_container_width=True)

        # =====================================================
        # DOWNLOAD CSV
        # =====================================================
        csv_all = hasil.to_csv(index=False).encode("utf-8")
        st.download_button(
            "💾 Download CSV Hasil Klaster",
            csv_all,
            "hasil_cluster.csv",
            "text/csv"
        )

        # =====================================================
        # STATISTIK TIAP CLUSTER
        # =====================================================
        st.subheader("📊 Statistik Ringkas Tiap Cluster")

        for i, cluster in enumerate(clusters):
            st.markdown(f"**Cluster {i+1}**")

            if not cluster:
                st.write("Tidak ada data.")
                continue

            values = pd.DataFrame([data for _, data in cluster])
            stats = values.agg(["mean", "min", "max"])
            st.dataframe(stats)

        # =====================================================
        # PROFIL KLASTER
        # =====================================================
        st.subheader("🧠 Profil Klaster Otomatis")

        for i, cluster in enumerate(clusters):
            jumlah = len(cluster)
            rata = sum(sum(data) for _, data in cluster) / (jumlah * len(cluster[0][1]))

            if rata >= 0.66:
                kategori = "Kerentanan Tinggi"
            elif rata >= 0.33:
                kategori = "Kerentanan Sedang"
            else:
                kategori = "Kerentanan Rendah"

            st.markdown(
                f"**Cluster {i+1}** terdiri dari **{jumlah} data** "
                f"dengan rata-rata nilai variabel **{rata:.2f}**, "
                f"menunjukkan **{kategori}**."
            )

        st.success("✅ Proses klasterisasi selesai.")
