import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA 
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
import io

st.set_page_config(page_title="Análisis de Segmentación de Clientes", layout="wide")
st.title("🧠 Análisis de Segmentación de Clientes")

# Cargar datos
st.header("📁 Datos Originales")
clientes = pd.read_csv("customer_data.csv")
st.dataframe(clientes.head())

# Info básica
buffer = io.StringIO()
clientes.info(buf=buffer)
st.subheader("🔍 Información del Dataset")
st.text(buffer.getvalue())

st.subheader("📊 Estadísticas Descriptivas")
st.dataframe(clientes.describe())

# Normalización
escalador = MinMaxScaler()
clientes_escalados = escalador.fit_transform(clientes[["Edad", "Ingresos Anuales (k$)", "Puntuación de Gasto (1-100)"]])

# PCA
pca = PCA(n_components=2)
pca_resultados = pca.fit_transform(clientes_escalados)

# SVD
U, sigma, VT = np.linalg.svd(clientes_escalados)
k = 2
svd_resultados = U[:, :k] * sigma[:k]

# KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_clusters = kmeans.fit_predict(clientes_escalados)

# Clustering jerárquico
linked = linkage(clientes_escalados, 'ward')

# Gráfico PCA
st.subheader("🎯 Dispersión PCA + KMeans")
fig1, ax1 = plt.subplots()
sns.scatterplot(x=pca_resultados[:, 0], y=pca_resultados[:, 1], hue=kmeans_clusters, palette='viridis', s=100, ax=ax1)
ax1.set_title("Gráfico de Dispersión PCA")
ax1.set_xlabel("Componente Principal 1")
ax1.set_ylabel("Componente Principal 2")
ax1.legend(title="Cluster")
st.pyplot(fig1)

# Dendrograma
st.subheader("🧬 Dendrograma de Clustering Jerárquico")
fig2, ax2 = plt.subplots(figsize=(10, 6))
dendrogram(linked, ax=ax2)
ax2.set_title("Dendrograma de Clustering Jerárquico")
ax2.set_xlabel("Índice de muestra")
ax2.set_ylabel("Distancia (Ward)")
ax2.axhline(y=10, color='r', linestyle='--')
st.pyplot(fig2)

# Análisis de Clusters
st.subheader("📌 Análisis de Clusters (Resumen)")
cluster_info = pd.DataFrame({
    'Cluster': kmeans_clusters,
    'Edad': clientes['Edad'],
    'Ingresos': clientes['Ingresos Anuales (k$)'],
    'Gasto': clientes['Puntuación de Gasto (1-100)']
})

for cluster in cluster_info['Cluster'].unique():
    cluster_data = cluster_info[cluster_info['Cluster'] == cluster]
    st.markdown(f"### 🧩 Cluster {cluster}")
    st.markdown(f"- Edad media: **{cluster_data['Edad'].mean():.0f}** años")
    st.markdown(f"- Ingresos medios: **{cluster_data['Ingresos'].mean():.2f}k$**")
    st.markdown(f"- Gasto medio: **{cluster_data['Gasto'].mean():.2f} puntos**")

# Bonus: Dendrograma Iris
st.subheader("🌸 Bonus: Dendrograma con Dataset Iris")
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
Z = linkage(X, 'ward')

fig3, ax3 = plt.subplots(figsize=(10, 6))
dendrogram(Z, orientation="top", labels=iris.target, distance_sort="descending", show_leaf_counts=True, ax=ax3)
ax3.set_title("Dendrograma del Agrupamiento Jerárquico - Iris")
ax3.set_xlabel("Índice de la muestra")
ax3.set_ylabel("Distancia")
st.pyplot(fig3)
