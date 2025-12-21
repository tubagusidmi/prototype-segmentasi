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

def warna_kategori(skor):
    if skor >= 0.75:
        return "red", "Sangat Tinggi"
    elif skor >= 0.60:
        return "orange", "Tinggi"
    elif skor >= 0.40:
        return "gold", "Sedang"
    else:
        return "green", "Sangat Sedang"

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
        # PCA GLOBAL
        # =====================================================
        st.subheader("📈 PCA Scatter Plot (Global)")

        pca = PCA(n_components=2)
        data_2d = pca.fit_transform(dataset)

        fig, ax = plt.subplots()
        for i in range(K):
            pts = data_2d[[j for j in range(len(labels)) if labels[j] == i]]
            ax.scatter(pts[:, 0], pts[:, 1], label=f"Cluster {i+1}")
        ax.legend()
        st.pyplot(fig)

        # =====================================================
        # HASIL DATA
        # =====================================================
        hasil = df.copy()
        hasil["Cluster"] = [l + 1 for l in labels]

        # =====================================================
        # PILIH CLUSTER
        # =====================================================
        st.divider()
        st.subheader("🎯 Analisis Cluster Terpilih")

        cluster_pilih = st.selectbox(
            "Pilih Cluster:",
            [f"Cluster {i+1}" for i in range(K)]
        )
        cluster_idx = int(cluster_pilih.split()[-1])
        df_cluster = hasil[hasil["Cluster"] == cluster_idx]

        # =====================================================
        # SKOR & KATEGORI
        # =====================================================
        skor = df_cluster.drop(columns=["Cluster"]).values.mean()
        warna, kategori = warna_kategori(skor)

        st.markdown(
            f"""
            ### 🧠 Ringkasan Cluster {cluster_idx}
            - 🎯 Jumlah Data: **{len(df_cluster)}**
            - 📊 Skor Kerentanan: **{skor:.2f}**
            - 🚦 Kategori: **:{warna}[{kategori}]**
            """
        )

        # =====================================================
        # ANGGOTA CLUSTER
        # =====================================================
        st.subheader("📋 Anggota Cluster Terpilih")
        st.dataframe(df_cluster.head(20), use_container_width=True)

        # =====================================================
        # PCA HIGHLIGHT
        # =====================================================
        st.subheader("📈 PCA Scatter Plot (Highlight Cluster)")

        fig2, ax2 = plt.subplots()
        for i in range(K):
            pts = data_2d[[j for j in range(len(labels)) if labels[j] == i]]
            if i + 1 == cluster_idx:
                ax2.scatter(pts[:, 0], pts[:, 1], s=120, label=f"Cluster {i+1}")
            else:
                ax2.scatter(pts[:, 0], pts[:, 1], alpha=0.2)
        ax2.legend()
        st.pyplot(fig2)

        # =====================================================
        # BAR KERENTANAN
        # =====================================================
        st.subheader("📊 Grafik Bar Tingkat Kerentanan")

        skor_cluster = []
        warna_bar = []

        for i, cluster in enumerate(clusters):
            if not cluster:
                s = 0
            else:
                s = sum(sum(d) for _, d in cluster) / (len(cluster) * dim)
            skor_cluster.append(s)
            warna_bar.append(warna_kategori(s)[0] if i + 1 == cluster_idx else "lightgray")

        fig3, ax3 = plt.subplots()
        ax3.bar([f"Cluster {i+1}" for i in range(K)], skor_cluster, color=warna_bar)
        ax3.set_ylim(0, 1)
        st.pyplot(fig3)

        # =====================================================
        # KARAKTERISTIK
        # =====================================================
        st.subheader("🧬 Karakteristik Cluster")
        st.dataframe(df_cluster.drop(columns=["Cluster"]).mean().to_frame("Rata-rata"))

        # =====================================================
        # PENYEBAB & SOLUSI
        # =====================================================
        st.subheader("⚠️ Penyebab Potensial Anak Putus Sekolah")
        st.markdown("""
        - Kondisi sosial ekonomi keluarga yang terbatas  
        - Rendahnya pendidikan orang tua  
        - Anak harus bekerja membantu keluarga  
        - Lingkungan tempat tinggal kurang mendukung pendidikan  
        """)

        st.subheader("🛠️ Solusi dan Rekomendasi Kebijakan")
        st.markdown("""
        - Bantuan pendidikan tepat sasaran  
        - Program pendampingan keluarga rentan  
        - Akses pendidikan nonformal  
        - Kolaborasi sekolah, pemerintah, dan masyarakat  
        """)

        st.success("✅ Analisis cluster selesai dan siap digunakan.")
