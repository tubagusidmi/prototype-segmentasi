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
    "Aplikasi web interaktif untuk melakukan segmentasi anak putus sekolah "
    "menggunakan algoritma **K-Means Clustering**."
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
# FUNGSI K-MEANS (AMAN)
# =====================================================
def euclidean(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(min(len(a), len(b)))))

def init_centroids(data, k):
    return [data[i][:] for i in random.sample(range(len(data)), k)]

def assign_clusters(data, centroids):
    clusters = [[] for _ in centroids]
    labels = []
    for idx, point in enumerate(data):
        distances = [euclidean(point, c) for c in centroids]
        cidx = distances.index(min(distances))
        clusters[cidx].append((idx + 1, point))
        labels.append(cidx)
    return clusters, labels

def compute_centroids(clusters, dim):
    centroids = []
    for cluster in clusters:
        if not cluster:
            centroids.append([0] * dim)
        else:
            centroids.append(
                [sum(p[1][i] for p in cluster) / len(cluster) for i in range(dim)]
            )
    return centroids

def kategori_kerentanan(skor):
    if skor >= 0.75:
        return "🔴 Sangat Tinggi (Sangat Rentan)"
    elif skor >= 0.60:
        return "🟠 Tinggi (Mendekati Rentan)"
    elif skor >= 0.40:
        return "🟡 Rendah"
    else:
        return "🟢 Sangat Rendah (Lebih Baik)"

# =====================================================
# PROSES UTAMA
# =====================================================
if uploaded_file is not None:

    try:
        df_raw = pd.read_csv(uploaded_file, header=None)
    except:
        st.error("❌ Gagal membaca file CSV.")
        st.stop()

    dataset = []
    for _, row in df_raw.iterrows():
        try:
            dataset.append([float(x) for x in row.tolist()])
        except:
            continue

    if len(dataset) == 0:
        st.error("❌ Dataset kosong atau tidak valid.")
        st.stop()

    dim = len(dataset[0])
    dataset = [row for row in dataset if len(row) == dim]

    df = pd.DataFrame(dataset)
    st.success(f"📁 Dataset dimuat: {len(dataset)} data, {dim} variabel")

    if st.button("🚀 Proses K-Means"):

        random.seed(42)
        centroids = init_centroids(dataset, K)

        for _ in range(MAX_ITER):
            clusters, labels = assign_clusters(dataset, centroids)
            new_centroids = compute_centroids(clusters, dim)
            if centroids == new_centroids:
                break
            centroids = new_centroids

        # =====================================================
        # PCA TRANSFORM
        # =====================================================
        pca = PCA(n_components=2)
        data_2d = pca.fit_transform(dataset)
        centroids_2d = pca.transform(centroids)

        # =====================================================
        # PILIH CLUSTER
        # =====================================================
        st.divider()
        st.subheader("🎯 Analisis Cluster Terpilih")

        cluster_idx = st.selectbox(
            "Pilih Cluster:",
            options=list(range(1, K + 1))
        )

        df["Cluster"] = [l + 1 for l in labels]
        df_cluster = df[df["Cluster"] == cluster_idx]

        skor = df_cluster.drop(columns=["Cluster"]).values.mean()
        kategori = kategori_kerentanan(skor)

        st.markdown(
            f"""
            ### 🧠 Ringkasan Cluster {cluster_idx}
            - 👥 Jumlah Anggota: **{len(df_cluster)}**
            - 📊 Skor Rata-rata: **{skor:.2f}**
            - 🚦 Tingkat Kerentanan: **{kategori}**
            """
        )

        # =====================================================
        # ANGGOTA CLUSTER
        # =====================================================
        st.subheader("📋 Anggota Cluster Terpilih")
        st.dataframe(df_cluster.head(20), use_container_width=True)

        # =====================================================
        # PCA SCATTER + CENTROID
        # =====================================================
        st.subheader("📈 PCA Scatter Plot (Cluster & Centroid)")

        fig, ax = plt.subplots()
        for i in range(K):
            idxs = [j for j, l in enumerate(labels) if l == i]
            pts = data_2d[idxs]

            if i + 1 == cluster_idx:
                ax.scatter(pts[:, 0], pts[:, 1], s=120, label=f"Cluster {i+1}")
            else:
                ax.scatter(pts[:, 0], pts[:, 1], alpha=0.2)

        # Centroid
        ax.scatter(
            centroids_2d[:, 0],
            centroids_2d[:, 1],
            marker="X",
            s=300,
            c="black",
            label="Centroid"
        )

        for i, (x, y) in enumerate(centroids_2d):
            ax.text(x, y, f"C{i+1}", fontsize=11, fontweight="bold")

        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.legend()
        st.pyplot(fig)

        # =====================================================
        # KARAKTERISTIK DESKRIPTIF
        # =====================================================
        st.subheader("🧬 Karakteristik Cluster")
        st.markdown("""
        Cluster ini menunjukkan pola kondisi sosial ekonomi yang relatif serupa,
        terutama pada tingkat pendidikan, penghasilan keluarga, dan kondisi tempat tinggal.
        Anak pada cluster ini berpotensi menghadapi hambatan pendidikan yang perlu
        ditangani secara khusus sesuai tingkat kerentanannya.
        """)

        # =====================================================
        # PENYEBAB & SOLUSI
        # =====================================================
        st.subheader("⚠️ Penyebab Potensial Anak Putus Sekolah")
        st.markdown("""
        - Keterbatasan ekonomi keluarga  
        - Rendahnya dukungan pendidikan dari lingkungan keluarga  
        - Tekanan untuk bekerja membantu orang tua  
        - Akses pendidikan yang kurang memadai  
        """)

        st.subheader("🛠️ Solusi dan Rekomendasi Kebijakan")
        st.markdown("""
        - Bantuan pendidikan berbasis cluster kerentanan  
        - Program pendampingan keluarga rentan  
        - Penguatan pendidikan nonformal dan kejar paket  
        - Kolaborasi pemerintah, sekolah, dan masyarakat  
        """)

        st.success("✅ Analisis cluster berhasil dan siap dipresentasikan.")
