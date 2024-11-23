import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

arquivo = 'planets.csv'
dados = pd.read_csv(arquivo)

nomes_planetas = dados["Planet"]

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
cov_matriz = np.cov(dados_padronizados, rowvar=False)
print("Matriz de Covariância:\n", cov_matriz)

# PCA
pca = PCA(n_components=2)
resultado_pca = pca.fit_transform(dados_padronizados)

# Autovalores | Autovetores
autovalores = pca.explained_variance_
autovetores = pca.components_

print("\nAutovalores:\n", autovalores)
print("\nAutovetores:\n", autovetores)

plt.figure(figsize=(8, 6))
plt.scatter(resultado_pca[:, 0], resultado_pca[:, 1], c='purple', edgecolor='k')
for i, planet in enumerate(nomes_planetas):
    plt.text(resultado_pca[i, 0] + 0.1, resultado_pca[i, 1] + 0.1, planet, fontsize=10)
plt.title("PCA - Planetas do Sistema Solar")
plt.xlabel("Componente X")
plt.ylabel("Componente Y")
plt.grid()
plt.show()
