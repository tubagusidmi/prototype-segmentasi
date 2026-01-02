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
# HEADER
# =====================================================
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

# 🔴 TOMBOL RESET (WAJIB ADA)
if st.sidebar.button("🔄 Reset & Hitung Ulang"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.experimental_rerun()

# =====================================================
# FUNGSI K-MEANS
# =====================================================
def euclidean(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(len(a))))

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

# =====================================================
# LOAD DATASET & PROSES K-MEANS
# =====================================================
if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file, header=None)
    dataset = df_raw.values.tolist()
    dim = len(dataset[0])

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

            st.session_state.df = pd.DataFrame(dataset)
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
    data_2d = st.session_state.data_2d
    centroids_2d = st.session_state.centroids_2d

    hasil = df.copy()
    hasil["Cluster"] = [l + 1 for l in labels]

    colors = ['green', 'red', 'orange', 'cyan', 'purple', 'brown', 'pink', 'blue'][:K]

    st.subheader("📈 PCA Scatter Plot Semua Cluster")
    fig_pca, ax_pca = plt.subplots()

    for i in range(K):
        pts = data_2d[[j for j in range(len(labels)) if labels[j] == i]]
        ax_pca.scatter(
            pts[:, 0],
            pts[:, 1],
            color=colors[i],
            alpha=0.5,
            label=f"Cluster {i+1}"
        )

    # TITIK CENTROID
    ax_pca.scatter(
        centroids_2d[:, 0],
        centroids_2d[:, 1],
        marker="*",
        s=350,
        c="black",
        label="Centroid"
    )

    # 🔥 LABEL CENTROID (C1, C2, C3, ...)
    for i, (x, y) in enumerate(centroids_2d):
        ax_pca.text(
            x + 0.05,
            y + 0.05,
            f"C{i+1}",
            fontsize=13,
            fontweight="bold",
            color="black",
            bbox=dict(
                facecolor="white",
                edgecolor="black",
                boxstyle="round,pad=0.3"
            )
        )

    ax_pca.set_xlabel("PCA 1")
    ax_pca.set_ylabel("PCA 2")
    ax_pca.set_title("Visualisasi PCA dengan Label Centroid")
    ax_pca.legend()
    st.pyplot(fig_pca)
