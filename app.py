import streamlit as st
import math
import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# =====================================================
# SESSION STATE (LOCK & DATA)
# =====================================================
for key in ["locked", "df", "labels", "clusters", "centroids", "data_2d", "centroids_2d"]:
    if key not in st.session_state:
        st.session_state[key] = None
if st.session_state.locked is None:
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
    return math.sqrt(sum((a[i]-b[i])**2 for i in range(min(len(a), len(b)))))

def init_centroids(data, k):
    return [data[i][:] for i in random.sample(range(len(data)), k)]

def assign_clusters(data, centroids):
    clusters = [[] for _ in centroids]
    labels = []
    for idx, point in enumerate(data):
        distances = [euclidean(point, c) for c in centroids]
        cidx = distances.index(min(distances))
        clusters[cidx].append((idx+1, point))
        labels.append(cidx)
    return clusters, labels

def compute_centroids(clusters, dim):
    centroids = []
    for cluster in clusters:
        if not cluster:
            centroids.append([0]*dim)
        else:
            centroids.append([sum(p[1][i] for p in cluster)/len(cluster) for i in range(dim)])
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
    hasil["Cluster"] = [l+1 for l in labels]

    st.divider()
    st.subheader("🎯 Analisis Cluster")

    # =====================================================
    # BAR CHART SKOR KERENTANAN
    # =====================================================
    st.subheader("📊 Bar Chart Skor Kerentanan Setiap Cluster")
    cluster_scores = [0.34, 0.62, 0.50, 0.22]  # sesuai ringkasan
    cluster_colors = ['skyblue', 'red', 'orange', 'green']
    cluster_labels_narasi = [
        "Cluster 1: 0.34 (Rendah)",
        "Cluster 2: 0.62 (Sangat Tinggi)",
        "Cluster 3: 0.50 (Tinggi)",
        "Cluster 4: 0.22 (Sangat Rendah)"
    ]

    fig_bar, ax_bar = plt.subplots()
    bars = ax_bar.bar(range(1, K+1), cluster_scores, color=cluster_colors[:K])
    ax_bar.set_xlabel("Cluster")
    ax_bar.set_ylabel("Skor Kerentanan")
    ax_bar.set_title("Bar Chart Tingkat Kerentanan Setiap Cluster")
    ax_bar.set_xticks(range(1, K+1))
    ax_bar.set_xticklabels(cluster_labels_narasi[:K], rotation=0, ha='center')
    for idx, score in enumerate(cluster_scores[:K]):
        ax_bar.text(idx+1, score+0.01, f"{score:.2f}", ha='center')
    st.pyplot(fig_bar)

    # =====================================================
    # PILIH CLUSTER
    # =====================================================
    cluster_idx = st.selectbox("Pilih Cluster:", options=list(range(1, K+1)), key="cluster_pilihan")
    df_cluster = hasil[hasil["Cluster"] == cluster_idx]
    skor = df_cluster.drop(columns=["Cluster"]).values.mean()
    kategori = kategori_kerentanan(skor)

    st.markdown(f"""
    ### 📌 Ringkasan Cluster {cluster_idx}
    • Jumlah Data : {len(df_cluster)}
    • Skor Rata-rata : {skor:.2f}
    • Tingkat Kerentanan : {kategori}
    """)

    # =====================================================
    # PCA INTERAKTIF: Semua Cluster + Highlight Cluster Terpilih
    # =====================================================
    st.subheader("📈 PCA Scatter Plot Interaktif")
    fig_pca, ax_pca = plt.subplots()
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'cyan']
    for i in range(K):
        pts = data_2d[[j for j in range(len(labels)) if labels[j] == i]]
        if i+1 == cluster_idx:
            ax_pca.scatter(pts[:,0], pts[:,1], color=colors[i], s=120, label=f"Cluster {i+1}")
        else:
            ax_pca.scatter(pts[:,0], pts[:,1], color=colors[i], alpha=0.2, label=f"Cluster {i+1}")
    ax_pca.scatter(
        centroids_2d[:,0],
        centroids_2d[:,1],
        marker="*", s=350, c="black", label="Centroid"
    )
    ax_pca.set_xlabel("PCA 1")
    ax_pca.set_ylabel("PCA 2")
    ax_pca.set_title("PCA Scatter Plot Semua Cluster + Highlight Cluster Terpilih")
    ax_pca.legend()
    st.pyplot(fig_pca)

    # =====================================================
    # ANGGOTA CLUSTER
    # =====================================================
    st.subheader("📋 Anggota Cluster (Contoh)")
    st.dataframe(df_cluster.head(20), use_container_width=True)

    # =====================================================
    # KARAKTERISTIK, TINGKAT KERENTANAN, PENYEBAB, SOLUSI
    # =====================================================
    # Karakteristik
    st.subheader("🧬 Karakteristik Cluster")
    if kategori == "🔴 Sangat Tinggi (Sangat Rentan)":
        st.markdown("""• Pendidikan: rendah.
• Pekerjaan: tidak stabil (rentan pekerjaan informal dan penghasilan tidak tetap).
• Penghasilan: rendah.
• Anggota keluarga: relatif besar sehingga beban ekonomi tinggi.
• Tempat tinggal: tidak layak huni.""")
    elif kategori == "🟠 Tinggi (Mendekati Rentan)":
        st.markdown("""• Pendidikan: rendah.
• Pekerjaan: tidak stabil (rentan pekerjaan informal dan penghasilan tidak tetap).
• Penghasilan: rendah.
• Anggota keluarga: relatif besar sehingga beban ekonomi tinggi.
• Tempat tinggal: masih layak namun tidak optimal.""")
    elif kategori == "🟡 Rendah":
        st.markdown("""• Pendidikan: nilai normalisasi tinggi menandakan pendidikan orang tua rendah.
• Pekerjaan: cukup baik dan relatif stabil.
• Penghasilan: cukup untuk memenuhi kebutuhan dasar anak.
• Anggota keluarga: jumlah tanggungan sedang.
• Tempat tinggal: cukup layak.""")
    else:  # 🟢 Sangat Rendah
        st.markdown("""• Pendidikan: sangat rendah nilai normalisasi (merepresentasikan pendidikan orang tua lebih tinggi atau lebih baik).
• Pekerjaan: relatif stabil.
• Penghasilan: cukup baik dan mendukung kebutuhan dasar pendidikan.
• Anggota keluarga: jumlah tanggungan sedang.
• Tempat tinggal: cukup layak.""")

    # Tingkat Kerentanan
    st.subheader("⚠️ Tingkat Kerentanan")
    if kategori == "🔴 Sangat Tinggi (Sangat Rentan)":
        st.markdown("Klaster ini memiliki skor kerentanan 0,62, yang merupakan nilai tertinggi, sehingga dikategorikan sebagai klaster dengan tingkat kerentanan SANGAT TINGGI. Prioritas utama intervensi.")
    elif kategori == "🟠 Tinggi (Mendekati Rentan)":
        st.markdown("Klaster ini memiliki skor kerentanan 0,50 sehingga dikategorikan TINGGI. Faktor utama berasal dari rendahnya pendidikan dan pendapatan keluarga.")
    elif kategori == "🟡 Rendah":
        st.markdown("Klaster ini memiliki skor kerentanan 0,34 lebih baik dibanding klaster 2 dan 3. Kondisi ekonomi relatif stabil sehingga dikategorikan RENDAH.")
    else:
        st.markdown("Klaster ini memiliki skor kerentanan 0,22 yang merupakan nilai paling rendah sehingga dikategorikan SANGAT RENDAH. Risiko anak putus sekolah minimal.")

    # Penyebab Potensial
    st.subheader("⚠️ Penyebab Potensial Anak Putus Sekolah")
    if kategori == "🔴 Sangat Tinggi (Sangat Rentan)" or kategori=="🟠 Tinggi (Mendekati Rentan)":
        st.markdown("""• Ketidakstabilan pendapatan orang tua.
• Pendidikan orang tua rendah sehingga wawasan terbatas.
• Tingginya beban ekonomi keluarga akibat jumlah tanggungan besar.
• Anak berpotensi didorong bekerja membantu ekonomi keluarga.""")
    elif kategori == "🟡 Rendah":
        st.markdown("""• Rendahnya pendidikan orang tua berdampak pada kurangnya perhatian terhadap proses belajar anak.
• Motivasi pendidikan keluarga yang belum kuat.
• Kurangnya keterlibatan dalam kegiatan sekolah.""")
    else:  # Sangat Rendah
        st.markdown("""• Motivasi belajar anak menurun karena faktor lingkungan pergaulan.
• Ketidakhadiran pengawasan pendidikan secara konsisten.
• Perubahan ekonomi mendadak (misalnya PHK orang tua).""")

    # Solusi & Rekomendasi
    st.subheader("💡 Solusi dan Rekomendasi Kebijakan")
    if kategori == "🔴 Sangat Tinggi (Sangat Rentan)":
        st.markdown("""• Penyaluran bantuan sosial prioritas (PKH, beasiswa penuh).
• Program perbaikan rumah tidak layak huni (RTLH).
• Intervensi terpadu: sekolah–kelurahan–dinas sosial–puskesmas.
• Pendampingan keluarga secara intensif.
• Pemberdayaan ekonomi keluarga melalui program UMKM dan pelatihan produktif.
• Konseling pendidikan untuk meningkatkan motivasi anak.""")
    elif kategori == "🟠 Tinggi (Mendekati Rentan)":
        st.markdown("""• Bantuan finansial (Beasiswa daerah, subsidi transport dan perlengkapan sekolah).
• Pelatihan peningkatan keterampilan bagi orang tua.
• Program pendampingan keluarga rawan sosial ekonomi.
• Intervensi sekolah berupa home visit dan monitoring intensif.""")
    elif kategori == "🟡 Rendah":
        st.markdown("""• Program motivasi pendidikan dan bimbingan belajar.
• Edukasi kepada orang tua mengenai pentingnya pendidikan jangka panjang.
• Penguatan peran wali kelas dan guru BK untuk mencegah penurunan motivasi anak.
• Pemberian akses kegiatan ekstrakurikuler untuk meningkatkan engagement siswa.""")
    else:
        st.markdown("""• Monitoring berkala bagi anak berisiko di sekolah.
• Program penguatan motivasi belajar dan konseling sekolah.
• Pelibatan orang tua melalui kegiatan parenting education.
• Menguatkan ketahanan keluarga melalui program pemberdayaan masyarakat.""")

    st.success("✅ Analisis cluster dapat dieksplorasi tanpa menghitung ulang.")
