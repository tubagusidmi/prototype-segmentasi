import streamlit as st
import math
import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# =====================================================
# SESSION STATE (LOCK & DATA)
# =====================================================
if "locked" not in st.session_state:
    st.session_state.locked = False

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

K = st.sidebar.slider(
    "Jumlah Cluster (K)",
    min_value=2,
    max_value=8,
    value=4,
    disabled=st.session_state.locked
)

MAX_ITER = 100

if st.session_state.locked:
    st.sidebar.warning("🔒 Hasil klaster sudah dikunci")

# =====================================================
# FUNGSI K-MEANS
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
            centroids.append([sum(p[1][i] for p in cluster)/len(cluster) for i in range(dim)])
    return centroids


def kategori_kerentanan(skor):
    if skor >= 0.75:
        return "Sangat Tinggi"
    elif skor >= 0.60:
        return "Tinggi"
    elif skor >= 0.40:
        return "Rendah"
    else:
        return "Sangat Rendah"


def narasi_kerentanan(skor):
    if skor >= 0.60:
        return """📊 Karakteristik
• Pendidikan: rendah.
• Pekerjaan: tidak stabil (rentan pekerjaan informal dan penghasilan tidak tetap).
• Penghasilan: rendah.
• Anggota keluarga: relatif besar sehingga beban ekonomi tinggi.
• Tempat tinggal: tidak layak huni.
⚠️ Tingkat Kerentanan
Klaster ini memiliki skor kerentanan tinggi (SANGAT TINGGI). Prioritas utama intervensi.
💡 Solusi dan Rekomendasi Kebijakan
• Penyaluran bantuan sosial prioritas.
• Program perbaikan rumah tidak layak huni.
• Intervensi terpadu sekolah–kelurahan–dinas sosial.
• Pendampingan keluarga oleh tenaga kesejahteraan sosial.
• Pemberdayaan ekonomi keluarga melalui program UMKM.
⚠️ Penyebab Potensial Anak Putus Sekolah
• Ketidakstabilan pendapatan orang tua.
• Pendidikan orang tua rendah.
• Tingginya beban ekonomi keluarga.
• Anak berpotensi bekerja membantu ekonomi keluarga.
"""
    elif skor >= 0.50:
        return """📊 Karakteristik
• Pendidikan: rendah.
• Pekerjaan: tidak stabil (rentan pekerjaan informal dan penghasilan tidak tetap).
• Penghasilan: rendah.
• Anggota keluarga: relatif besar sehingga beban ekonomi tinggi.
• Tempat tinggal: masih layak namun tidak optimal.
⚠️ Tingkat Kerentanan
Klaster ini memiliki skor kerentanan TINGGI.
💡 Solusi dan Rekomendasi Kebijakan
• Bantuan finansial (beasiswa, subsidi transport).
• Pelatihan keterampilan bagi orang tua.
• Pendampingan keluarga rawan sosial ekonomi.
• Home visit dan monitoring intensif.
⚠️ Penyebab Potensial Anak Putus Sekolah
• Ketidakstabilan pendapatan orang tua.
• Pendidikan orang tua rendah.
• Tingginya beban ekonomi keluarga.
• Anak berpotensi bekerja membantu ekonomi keluarga.
"""
    elif skor >= 0.40:
        return """📊 Karakteristik
• Pendidikan: nilai normalisasi tinggi menandakan pendidikan orang tua rendah.
• Pekerjaan: cukup baik dan relatif stabil.
• Penghasilan: cukup untuk memenuhi kebutuhan dasar anak.
• Anggota keluarga: jumlah tanggungan sedang.
• Tempat tinggal: cukup layak.
⚠️ Tingkat Kerentanan
Klaster ini memiliki skor kerentanan RENDAH.
💡 Solusi dan Rekomendasi Kebijakan
• Program motivasi pendidikan dan bimbingan belajar.
• Edukasi orang tua mengenai pentingnya pendidikan.
• Penguatan peran wali kelas dan guru BK.
• Akses kegiatan ekstrakurikuler untuk meningkatkan engagement siswa.
⚠️ Penyebab Potensial Anak Putus Sekolah
• Rendahnya pendidikan orang tua berdampak pada kurangnya perhatian terhadap belajar anak.
• Motivasi pendidikan keluarga belum kuat.
• Kurangnya keterlibatan kegiatan sekolah.
"""
    else:
        return """📊 Karakteristik
• Pendidikan: sangat rendah (merepresentasikan pendidikan orang tua lebih baik).
• Pekerjaan: relatif stabil.
• Penghasilan: cukup baik dan mendukung kebutuhan dasar pendidikan.
• Anggota keluarga: jumlah tanggungan sedang.
• Tempat tinggal: cukup layak.
⚠️ Tingkat Kerentanan
Klaster ini memiliki skor kerentanan SANGAT RENDAH.
💡 Solusi dan Rekomendasi Kebijakan
• Monitoring berkala bagi anak berisiko di sekolah.
• Program penguatan motivasi belajar dan konseling.
• Pelibatan orang tua melalui parenting education.
• Penguatan ketahanan keluarga melalui pemberdayaan masyarakat.
⚠️ Penyebab Potensial Anak Putus Sekolah
• Motivasi belajar anak menurun karena lingkungan.
• Ketidakhadiran pengawasan pendidikan konsisten.
• Perubahan ekonomi mendadak (misal PHK orang tua).
"""

# =====================================================
# LOAD DATASET
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

    # =====================================================
    # PROSES K-MEANS
    # =====================================================
    if not st.session_state.locked:
        if st.button("🚀 Proses K-Means"):

            random.seed(42)
            centroids = init_centroids(dataset, K)

            for _ in range(MAX_ITER):
                clusters, labels = assign_clusters(dataset, centroids)
                new_centroids = compute_centroids(clusters, dim)
                if centroids == new_centroids:
                    break
                centroids = new_centroids

            # PCA
            pca = PCA(n_components=2)
            data_2d = pca.fit_transform(dataset)
            centroids_2d = pca.transform(centroids)

            # SIMPAN KE SESSION STATE (LOCK)
            st.session_state.df = df
            st.session_state.labels = labels
            st.session_state.clusters = clusters
            st.session_state.centroids = centroids
            st.session_state.data_2d = data_2d
            st.session_state.centroids_2d = centroids_2d
            st.session_state.locked = True

            st.success("✅ Proses K-Means selesai dan hasil dikunci")

# =====================================================
# TAMPILKAN HASIL
# =====================================================
if st.session_state.locked:
    df = st.session_state.df
    labels = st.session_state.labels
    clusters = st.session_state.clusters
    data_2d = st.session_state.data_2d
    centroids_2d = st.session_state.centroids_2d

    hasil = df.copy()
    hasil["Cluster"] = [l + 1 for l in labels]

    st.divider()
    st.subheader("🎯 Analisis Cluster")

    # =====================================================
    # BAR CHART SKOR KERENTANAN
    # =====================================================
    st.subheader("📊 Bar Chart Skor Kerentanan Setiap Cluster")
    cluster_scores = []
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'cyan']  # sampai K=8
    narasi_scores = []

    for i in range(K):
        df_c = hasil[hasil["Cluster"] == i+1]
        score_c = df_c.drop(columns=["Cluster"]).values.mean()
        cluster_scores.append(score_c)
        kategori = kategori_kerentanan(score_c)
        narasi_scores.append(f"Cluster {i+1}\nScore: {score_c:.2f} ({kategori})")

    fig_bar, ax_bar = plt.subplots()
    bars = ax_bar.bar(range(1, K+1), cluster_scores, color=colors[:K])
    ax_bar.set_xticks(range(1, K+1))
    ax_bar.set_xticklabels(narasi_scores, rotation=0)
    ax_bar.set_ylabel("Skor Kerentanan")
    ax_bar.set_title("Bar Chart Tingkat Kerentanan Setiap Cluster")
    for idx, score in enumerate(cluster_scores):
        ax_bar.text(idx+1, score+0.01, f"{score:.2f}", ha='center')
    st.pyplot(fig_bar)

    # =====================================================
    # PILIH CLUSTER
    # =====================================================
    cluster_idx = st.selectbox(
        "Pilih Cluster:",
        options=list(range(1, K + 1)),
        key="cluster_pilihan"
    )

    df_cluster = hasil[hasil["Cluster"] == cluster_idx]
    skor = df_cluster.drop(columns=["Cluster"]).values.mean()
    kategori = kategori_kerentanan(skor)

    st.markdown(f"""
    ### 📌 Ringkasan Cluster {cluster_idx}
    - Jumlah Data : **{len(df_cluster)}**
    - Skor Rata-rata : **{skor:.2f}**
    - Tingkat Kerentanan : **{kategori}**
    """)

    # =====================================================
    # PCA SCATTER PLOT INTERAKTIF
    # =====================================================
    st.subheader("📈 PCA Scatter Plot Interaktif (Semua Cluster + Highlight)")

    fig_pca, ax_pca = plt.subplots()
    for i in range(K):
        pts = data_2d[[j for j in range(len(labels)) if labels[j] == i]]
        if i + 1 == cluster_idx:
            ax_pca.scatter(pts[:,0], pts[:,1], s=150, color=colors[i], edgecolor='black', label=f'Cluster {i+1} (Highlight)')
        else:
            ax_pca.scatter(pts[:,0], pts[:,1], s=60, color=colors[i], alpha=0.25, label=f'Cluster {i+1}')

    ax_pca.scatter(
        centroids_2d[:,0],
        centroids_2d[:,1],
        marker="*",
        s=300,
        c="black",
        label="Centroid"
    )

    ax_pca.set_xlabel("PCA 1")
    ax_pca.set_ylabel("PCA 2")
    ax_pca.set_title("PCA Scatter Plot Semua Cluster + Highlight Cluster Terpilih")
    ax_pca.legend()
    st.pyplot(fig_pca)

    # =====================================================
    # NARASI BERDASARKAN TINGKAT KERENTANAN
    # =====================================================
    st.subheader("📌 Analisis Narasi Cluster")
    st.markdown(narasi_kerentanan(skor))
