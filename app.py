import streamlit as st
import math
import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

st.set_page_config(
    page_title="Prototipe Segmentasi Anak Putus Sekolah",
    layout="wide"
)

st.title("📊 Prototipe Segmentasi Anak Putus Sekolah")

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.header("⚙️ Pengaturan")

uploaded_file = st.sidebar.file_uploader("Upload Dataset CSV", type=["csv"])
K = st.sidebar.slider("Jumlah Cluster (K)", 2, 8, 4)
MAX_ITER = 100

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
        centroids.append(
            [sum(p[1][i] for p in cluster) / len(cluster) for i in range(dim)]
        )
    return centroids

# =====================================================
# LOAD & PROSES DATA
# =====================================================
if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file, header=None)

    dataset = df_raw.astype(float).values.tolist()
    dim = len(dataset[0])

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

        df = pd.DataFrame(dataset)
        df["Cluster"] = [l + 1 for l in labels]

        # =================================================
        # PILIH CLUSTER
        # =================================================
        cluster_idx = st.selectbox(
            "Pilih Cluster",
            sorted(df["Cluster"].unique())
        )

        df_cluster = df[df["Cluster"] == cluster_idx]

        st.markdown(f"""
### 📌 Ringkasan Cluster {cluster_idx}
- Jumlah Anggota: **{len(df_cluster)}**
""")

        # =================================================
        # PCA SCATTER
        # =================================================
        st.subheader("📈 PCA Scatter Plot")

        fig, ax = plt.subplots()
        for i in range(K):
            pts = data_2d[[j for j in range(len(labels)) if labels[j] == i]]
            ax.scatter(pts[:, 0], pts[:, 1], alpha=0.2)

        ax.scatter(
            centroids_2d[:, 0],
            centroids_2d[:, 1],
            marker="*",
            s=300,
            c="black"
        )

        for i, (x, y) in enumerate(centroids_2d):
            ax.text(
                x + 0.05,
                y + 0.05,
                f"C{i+1}",
                fontsize=12,
                fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="black")
            )

        st.pyplot(fig)

        # =================================================
        # TABEL ANGGOTA CLUSTER (FULL + SCROLL)
        # =================================================
        st.subheader(f"📋 Anggota Cluster {cluster_idx} (Lengkap)")

        st.dataframe(
            df_cluster.reset_index(drop=True),
            use_container_width=True,
            height=400
        )
