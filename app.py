import streamlit as st
import math
import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# =====================================================
# SESSION STATE
# =====================================================
if "locked" not in st.session_state:
    st.session_state.locked = False
if "df" not in st.session_state:
    st.session_state.df = None
if "labels" not in st.session_state:
    st.session_state.labels = None
if "clusters" not in st.session_state:
    st.session_state.clusters = None
if "centroids" not in st.session_state:
    st.session_state.centroids = None
if "data_2d" not in st.session_state:
    st.session_state.data_2d = None
if "centroids_2d" not in st.session_state:
    st.session_state.centroids_2d = None

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

# Fungsi untuk menentukan kategori kerentanan otomatis
def kategori_kerentanan(skor):
    if skor >= 0.60:
        return "Sangat Tinggi (Sangat Rentan)"
    elif skor >= 0.40:
        return "Tinggi (Rentan)"
    elif skor >= 0.30:
        return "Rendah (Sedikit Lebih Baik)"
    else:
        return "Sangat Rendah (Lebih Baik)"

# Fungsi deskripsi otomatis per skor
def deskripsi_cluster(skor):
    if skor >= 0.60:
        return {
            "Karakteristik": """
• Pendidikan: rendah
• Pekerjaan: tidak stabil
• Penghasilan: rendah
• Anggota keluarga: relatif besar
• Tempat tinggal: tidak layak huni
""",
            "Tingkat Kerentanan": f"Skor kerentanan {skor:.2f} → Sangat Tinggi (Sangat Rentan)",
            "Penyebab": """
• Ketidakstabilan pendapatan orang tua
• Pendidikan orang tua rendah
• Tingginya beban ekonomi keluarga
• Anak berpotensi bekerja membantu ekonomi keluarga
""",
            "Solusi": """
• Bantuan sosial prioritas (PKH, beasiswa)
• Perbaikan rumah tidak layak huni (RTLH)
• Intervensi terpadu sekolah–kelurahan–dinas sosial
• Pendampingan keluarga intensif
• Pemberdayaan ekonomi keluarga
• Konseling pendidikan
"""
        }
    elif skor >= 0.40:
        return {
            "Karakteristik": """
• Pendidikan: rendah
• Pekerjaan: tidak stabil
• Penghasilan: rendah
• Anggota keluarga: relatif besar
• Tempat tinggal: masih layak namun tidak optimal
""",
            "Tingkat Kerentanan": f"Skor kerentanan {skor:.2f} → Tinggi (Rentan)",
            "Penyebab": """
• Ketidakstabilan pendapatan orang tua
• Pendidikan orang tua rendah
• Tingginya beban ekonomi keluarga
• Anak berpotensi bekerja membantu ekonomi keluarga
""",
            "Solusi": """
• Bantuan finansial (beasiswa, subsidi sekolah)
• Pelatihan keterampilan orang tua
• Pendampingan keluarga rawan sosial ekonomi
• Intervensi sekolah: home visit & monitoring
"""
        }
    elif skor >= 0.30:
        return {
            "Karakteristik": """
• Pendidikan: nilai normalisasi tinggi (pendidikan orang tua rendah)
• Pekerjaan: cukup stabil
• Penghasilan: cukup untuk kebutuhan dasar
• Anggota keluarga: jumlah tanggungan sedang
• Tempat tinggal: cukup layak
""",
            "Tingkat Kerentanan": f"Skor kerentanan {skor:.2f} → Rendah (Sedikit Lebih Baik)",
            "Penyebab": """
• Pendidikan orang tua rendah → kurang perhatian belajar anak
• Motivasi pendidikan keluarga belum kuat
• Kurangnya keterlibatan dalam kegiatan sekolah
""",
            "Solusi": """
• Program motivasi pendidikan & bimbingan belajar
• Edukasi orang tua tentang pentingnya pendidikan
• Penguatan peran wali kelas & guru BK
• Akses kegiatan ekstrakurikuler
"""
        }
    else:
        return {
            "Karakteristik": """
• Pendidikan: sangat rendah (pendidikan orang tua lebih baik)
• Pekerjaan: relatif stabil
• Penghasilan: cukup & mendukung pendidikan
• Anggota keluarga: tanggungan sedang
• Tempat tinggal: cukup layak
""",
            "Tingkat Kerentanan": f"Skor kerentanan {skor:.2f} → Sangat Rendah (Lebih Baik)",
            "Penyebab": """
• Motivasi belajar anak menurun karena lingkungan
• Kurangnya pengawasan pendidikan
• Perubahan ekonomi mendadak (misalnya PHK orang tua)
""",
            "Solusi": """
• Monitoring berkala di sekolah
• Program penguatan motivasi belajar & konseling
• Pelibatan orang tua melalui parenting education
• Penguatan ketahanan keluarga & pemberdayaan masyarakat
"""
        }

# =====================================================
# LOAD DATASET DAN PROSES K-MEANS
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

            pca = PCA(n_components=2)
            data_2d = pca.fit_transform(dataset)
            centroids_2d = pca.transform(centroids)

            st.session_state.df = df
            st.session_state.labels = labels
            st.session_state.clusters = clusters
            st.session_state.centroids = centroids
            st.session_state.data_2d = data_2d
            st.session_state.centroids_2d = centroids_2d
            st.session_state.locked = True
            st.success("✅ Proses K-Means selesai dan hasil dikunci")

# =====================================================
# TAMPILKAN HASIL FINAL
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

    # Bar chart skor
    cluster_scores = []
    kategori_scores = []
    for i in range(1, K+1):
        df_c = hasil[hasil["Cluster"] == i]
        skor_c = df_c.drop(columns=["Cluster"]).values.mean()
        cluster_scores.append(skor_c)
        kategori_scores.append(kategori_kerentanan(skor_c))

    colors = ['green', 'red', 'orange', 'cyan', 'purple', 'brown', 'pink', 'blue'][:K]
    cluster_labels = [f"Cluster {i+1}: {cluster_scores[i]:.2f} ({kategori_scores[i]})" for i in range(K)]

    fig_bar, ax_bar = plt.subplots()
    ax_bar.bar(range(1, K+1), cluster_scores, color=colors)
    ax_bar.set_xticks(range(1, K+1))
    ax_bar.set_xticklabels(cluster_labels, rotation=15, ha='right')
    ax_bar.set_ylabel("Skor Kerentanan")
    ax_bar.set_title("Bar Chart Skor Kerentanan Setiap Cluster")
    st.pyplot(fig_bar)

    # Pilih cluster
    cluster_idx = st.selectbox("Pilih Cluster:", options=list(range(1, K+1)), key="cluster_pilihan")
    df_cluster = hasil[hasil["Cluster"] == cluster_idx]
    skor = df_cluster.drop(columns=["Cluster"]).values.mean()
    kategori = kategori_kerentanan(skor)
    deskripsi = deskripsi_cluster(skor)

    st.markdown(f"""
### 📌 Ringkasan Cluster {cluster_idx}
- Jumlah Data : **{len(df_cluster)}**
- Skor Rata-rata : **{skor:.2f}**
- Tingkat Kerentanan : **{kategori}**
""")

    # PCA gabungan + highlight
    st.subheader("📈 PCA Scatter Plot Semua Cluster (Highlight Terpilih)")
    fig_pca, ax_pca = plt.subplots()
    for i in range(K):
        pts = data_2d[[j for j in range(len(labels)) if labels[j] == i]]
        if i + 1 == cluster_idx:
            ax_pca.scatter(pts[:,0], pts[:,1], color=colors[i], s=120, label=f"Cluster {i+1} (Terpilih)")
        else:
            ax_pca.scatter(pts[:,0], pts[:,1], color=colors[i], alpha=0.15, s=60, label=f"Cluster {i+1}")
    ax_pca.scatter(centroids_2d[:,0], centroids_2d[:,1], marker="*", s=350, c="black", label="Centroid")
    ax_pca.set_xlabel("PCA 1")
    ax_pca.set_ylabel("PCA 2")
    ax_pca.set_title("PCA Scatter Plot Semua Cluster")
    ax_pca.legend()
    st.pyplot(fig_pca)

    # Anggota cluster
    st.subheader("📋 Anggota Cluster (Contoh)")
    st.dataframe(df_cluster.head(20), use_container_width=True)

    # Deskripsi lengkap
    st.subheader("🧬 Karakteristik Cluster")
    st.markdown(deskripsi["Karakteristik"])
    st.subheader("⚠️ Tingkat Kerentanan")
    st.markdown(deskripsi["Tingkat Kerentanan"])
    st.subheader("⚠️ Penyebab Potensial Anak Putus Sekolah")
    st.markdown(deskripsi["Penyebab"])
    st.subheader("💡 Solusi dan Rekomendasi Kebijakan")
    st.markdown(deskripsi["Solusi"])

    st.success("✅ Analisis cluster dapat dieksplorasi tanpa menghitung ulang.")
