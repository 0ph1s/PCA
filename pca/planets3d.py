import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import mpl_toolkits.mplot3d  # Import necessário para projeções 3D

arquivo = 'planets.csv'
dados = pd.read_csv(arquivo)

planet_names = dados["Planet"]

dados["Ring System?"] = dados["Ring System?"].map({"Yes": 1, "No": 0})
dados["Global Magnetic Field?"] = dados["Global Magnetic Field?"].map({"Yes": 1, "No": 0})
dados["Surface Pressure (bars)"] = pd.to_numeric(dados["Surface Pressure (bars)"], errors='coerce').fillna(0)

for col in dados.columns:
    if dados[col].dtype == "object" and col not in ["Planet", "Color"]:
        dados[col] = dados[col].str.replace(",", "", regex=False)
        dados[col] = pd.to_numeric(dados[col], errors='coerce')

dados_numericos = dados.drop(columns=["Planet", "Color"])

# Padronização
padronizador = StandardScaler()
dados_padronizados = padronizador.fit_transform(dados_numericos)

# Matriz Covariância
matriz_covariancia = np.cov(dados_padronizados.T)
print("Matriz de Covariância:")
print(matriz_covariancia)

# PCA
pca = PCA(n_components=3)
resultado_pca = pca.fit_transform(dados_padronizados)

# Autovalores | Autovetores
autovalores = pca.explained_variance_
autovetores = pca.components_

print("\nAutovalores:")
print(autovalores)

print("\nAutovetores:")
print(autovetores)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d", elev=48, azim=134)

ax.scatter(
    resultado_pca[:, 0],
    resultado_pca[:, 1],
    resultado_pca[:, 2],
    c='purple',
    edgecolor='k'
)

for i, planet in enumerate(planet_names):
    ax.text(
        resultado_pca[i, 0],
        resultado_pca[i, 1],
        resultado_pca[i, 2],
        planet,
        fontsize=10
    )

ax.set_title("PCA 3D - Planetas do Sistema Solar")
ax.set_xlabel("Componente X")
ax.set_ylabel("Componente Y")
ax.set_zlabel("Componente Z")

plt.show()
