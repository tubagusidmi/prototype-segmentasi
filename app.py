# =====================================================
# TAMPILKAN HASIL (SETELAH LOCK)
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

    # ==========================
    # Bar Chart Skor Kerentanan
    # ==========================
    st.subheader("📊 Bar Chart Skor Kerentanan Setiap Cluster")

    cluster_scores = []
    for i in range(K):
        df_c = hasil[hasil["Cluster"] == i+1]
        score_c = df_c.drop(columns=["Cluster"]).values.mean()
        cluster_scores.append(score_c)

    fig_bar, ax_bar = plt.subplots()
    ax_bar.bar(range(1, K+1), cluster_scores, color='skyblue')
    ax_bar.set_xlabel("Cluster")
    ax_bar.set_ylabel("Skor Kerentanan")
    ax_bar.set_title("Bar Chart Tingkat Kerentanan Setiap Cluster")
    for idx, score in enumerate(cluster_scores):
        ax_bar.text(idx+1, score+0.01, f"{score:.2f}", ha='center')
    st.pyplot(fig_bar)

    # ==========================
    # Pilih Cluster
    # ==========================
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
    # PCA SCATTER PLOT DENGAN LABEL CLUSTER
    # =====================================================
    st.subheader("📈 PCA Scatter Plot Semua Cluster (Label Cluster)")

    fig_pca, ax_pca = plt.subplots()
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'cyan']  # sampai K=8

    for i in range(K):
        pts = data_2d[[j for j in range(len(labels)) if labels[j] == i]]
        ax_pca.scatter(pts[:,0], pts[:,1], color=colors[i], label=f'Cluster {i+1}', s=60)

    ax_pca.scatter(
        centroids_2d[:,0],
        centroids_2d[:,1],
        marker="*",
        s=250,
        c="black",
        label="Centroid"
    )
    ax_pca.set_xlabel("PCA 1")
    ax_pca.set_ylabel("PCA 2")
    ax_pca.set_title("PCA Scatter Plot Semua Cluster")
    ax_pca.legend()
    st.pyplot(fig_pca)

    # =====================================================
    # PCA HIGHLIGHT + CENTROID ⭐
    # =====================================================
    st.subheader("📈 PCA Scatter Plot (Highlight Cluster)")

    fig, ax = plt.subplots()
    for i in range(K):
        pts = data_2d[[j for j in range(len(labels)) if labels[j] == i]]
        if i + 1 == cluster_idx:
            ax.scatter(pts[:, 0], pts[:, 1], s=120, label=f"Cluster {i+1}")
        else:
            ax.scatter(pts[:, 0], pts[:, 1], alpha=0.15)

    ax.scatter(
        centroids_2d[:, 0],
        centroids_2d[:, 1],
        marker="*",
        s=350,
        c="black",
        label="Centroid"
    )

    ax.legend()
    st.pyplot(fig)

    # =====================================================
    # ANGGOTA CLUSTER
    # =====================================================
    st.subheader("📋 Anggota Cluster (Contoh)")
    st.dataframe(df_cluster.head(20), use_container_width=True)

    # =====================================================
    # KARAKTERISTIK (DESKRIPSI)
    # =====================================================
    st.subheader("🧬 Karakteristik Cluster")
    st.markdown("""
    Cluster ini menunjukkan pola kondisi sosial ekonomi yang relatif serupa,
    di mana faktor pendidikan, pekerjaan orang tua, dan kondisi keluarga
    mempengaruhi keberlanjutan pendidikan anak.
    """)

    # =====================================================
    # PENYEBAB & SOLUSI
    # =====================================================
    st.subheader("⚠️ Penyebab Potensial Anak Putus Sekolah")
    st.markdown("""
    - Keterbatasan ekonomi keluarga  
    - Rendahnya pendidikan orang tua  
    - Anak harus membantu bekerja  
    - Lingkungan kurang mendukung pendidikan  
    """)

    st.subheader("🛠️ Solusi dan Rekomendasi Kebijakan")
    st.markdown("""
    - Bantuan pendidikan tepat sasaran  
    - Program pendampingan keluarga rentan  
    - Penguatan pendidikan nonformal  
    - Kolaborasi sekolah, pemerintah, dan masyarakat  
    """)

    st.success("✅ Analisis cluster dapat dieksplorasi tanpa menghitung ulang.")
