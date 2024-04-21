from sklearn.datasets import load_iris
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
from scipy.stats import kruskal
import pandas as pd

# Carregar o conjunto de dados Iris
iris_data = load_iris()

# 1. Média, mediana e desvio padrão do comprimento das pétalas de todas as espécies:
comprimento_petalas = iris_data.data[:, 2]  # Índice 2 corresponde ao comprimento das pétalas

# Média
media_comprimento_petalas = np.mean(comprimento_petalas)
print("Média do comprimento das pétalas:", media_comprimento_petalas)

# Mediana
mediana_comprimento_petalas = np.median(comprimento_petalas)
print("Mediana do comprimento das pétalas:", mediana_comprimento_petalas)

# Desvio padrão
desvio_padrao_comprimento_petalas = np.std(comprimento_petalas)
print("Desvio padrão do comprimento das pétalas:", desvio_padrao_comprimento_petalas)

# 2. Espécie de flor com a maior e menor largura média de sépala:
largura_media_sepala = [np.mean(iris_data.data[iris_data.target == i, 1]) for i in range(3)]

# Maior largura média de sépala
largura_media_maxima = max(largura_media_sepala)
indice_especie_maior = largura_media_sepala.index(largura_media_maxima)
print("Espécie com maior largura média de sépala:", iris_data.target_names[indice_especie_maior])

# Menor largura média de sépala
largura_media_minima = min(largura_media_sepala)
indice_especie_menor = largura_media_sepala.index(largura_media_minima)
print("Espécie com menor largura média de sépala:", iris_data.target_names[indice_especie_menor])

# 3. Correlação entre comprimento e largura das pétalas, e entre comprimento das pétalas e comprimento das sépalas:
comprimento_petalas = iris_data.data[:, 2]  # Comprimento das pétalas
largura_petalas = iris_data.data[:, 3]  # Largura das pétalas
comprimento_sepala = iris_data.data[:, 0]  # Comprimento das sépalas

# Correlação entre comprimento e largura das pétalas
correlacao_comprimento_largura_petalas = np.corrcoef(comprimento_petalas, largura_petalas)[0, 1]
print("Correlação entre comprimento e largura das pétalas:", correlacao_comprimento_largura_petalas)

# Correlação entre comprimento das pétalas e comprimento das sépalas
correlacao_comprimento_petalas_comprimento_sepala = np.corrcoef(comprimento_petalas, comprimento_sepala)[0, 1]
print("Correlação entre comprimento das pétalas e comprimento das sépalas:", correlacao_comprimento_petalas_comprimento_sepala)

# 4. Usando o algoritmo de agrupamento K-means, quantos clusters são ideais para representar as diferentes espécies de flores Iris?

# Lista para armazenar os valores de inércia
inertia = []

# Testar o número de clusters de 1 a 10
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(iris_data.data)
    inertia.append(kmeans.inertia_)

# Plotar o método do cotovelo
plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('Número de Clusters')
plt.ylabel('Inércia')
plt.title('Método do Cotovelo para Determinar o Número de Clusters')
plt.show()

# 5. Quais características (comprimento da sépala, largura da sépala, comprimento da pétala, largura da pétala)
# 5. são mais úteis para distinguir entre as diferentes espécies de flores?

# Instanciar o classificador da árvore de decisão
clf = DecisionTreeClassifier()

# Treinar o classificador
clf.fit(iris_data.data, iris_data.target)

# Extrair a importância das características
importancias_caracteristicas = clf.feature_importances_

# Nomes das características
nomes_caracteristicas = iris_data.feature_names

# Plotar a importância das características
plt.figure(figsize=(10, 6))
plt.barh(range(len(importancias_caracteristicas)), importancias_caracteristicas, tick_label=nomes_caracteristicas)
plt.xlabel('Importância das Características')
plt.ylabel('Características')
plt.title('Importância das Características para Distinguir Espécies de Flores Iris')
plt.show()

# 6. Usando o KDE, como é a distribuição da largura das pétalas para cada espécie de flor?

iris_df = pd.DataFrame(data=iris_data.data, columns=iris_data.feature_names)
iris_df['species'] = iris_data.target

# Mapear os números das espécies para seus respectivos nomes
iris_df['species'] = iris_df['species'].map({0: iris_data.target_names[0],
                                             1: iris_data.target_names[1],
                                             2: iris_data.target_names[2]})

# Plotar o KDE para a largura das pétalas de cada espécie
sns.displot(data=iris_df, x='petal width (cm)', hue='species', kind='kde', fill=True)
plt.title('Distribuição da Largura das Pétalas para Cada Espécie de Flor Iris')
plt.show()

# 7. Existem outliers nos dados? Como eles afetam a distribuição das espécies de flores? Plote os dados em scatter plots e dê destaque com cor vermelha para os outliers.

# Criar DataFrame com os dados

iris_df = pd.DataFrame(data=iris_data.data, columns=iris_data.feature_names)
iris_df['species'] = iris_data.target

# Mapear os números das espécies para seus respectivos nomes
iris_df['species'] = iris_df['species'].map({0: iris_data.target_names[0],
                                             1: iris_data.target_names[1],
                                             2: iris_data.target_names[2]})

# Plotar boxplots para cada característica
for feature in iris_data.feature_names:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='species', y=feature, data=iris_df)
    plt.title(f'Distribuição de {feature} por Espécie de Flor Iris')
    plt.xlabel('Espécie')
    plt.ylabel(feature)
    plt.show()
    
# 8. Qual é a probabilidade de encontrar uma flor Iris-versicolor com largura de pétala entre 1.5 e 2.0 cm?

# Identificar amostras da espécie Iris-versicolor
iris_versicolor_samples = iris_data.data[iris_data.target == 1]

# Contar o número de amostras com largura de pétala entre 1.5 e 2.0 cm
petal_width_15_to_20 = iris_versicolor_samples[(iris_versicolor_samples[:, 3] >= 1.5) & (iris_versicolor_samples[:, 3] <= 2.0)]
count_15_to_20 = len(petal_width_15_to_20)

# Número total de amostras da espécie Iris-versicolor
total_versicolor_samples = len(iris_versicolor_samples)

# Calcular a probabilidade
probability_15_to_20 = count_15_to_20 / total_versicolor_samples
print("Probabilidade de encontrar uma flor Iris-versicolor com largura de pétala entre 1.5 e 2.0 cm:", probability_15_to_20)

# 9. Como as medidas das sépalas se comparam entre as espécies de flores?
# 9. Existe uma diferença significativa?10. Existe uma relação entre o comprimento das
# 9. pétalas e o comprimento das sépalas em uma determinada espécie de flor?

# Separar as medidas das sépalas por espécie
sepal_length_setosa = iris_data.data[iris_data.target == 0, 0]  # Comprimento das sépalas da espécie setosa
sepal_length_versicolor = iris_data.data[iris_data.target == 1, 0]  # Comprimento das sépalas da espécie versicolor
sepal_length_virginica = iris_data.data[iris_data.target == 2, 0]  # Comprimento das sépalas da espécie virginica

# Executar o teste de Kruskal-Wallis
h_statistic, p_value = kruskal(sepal_length_setosa, sepal_length_versicolor, sepal_length_virginica)

# Interpretar os resultados
print("Estatística H:", h_statistic)
print("Valor-p:", p_value)

if p_value < 0.05:
    print("Existe uma diferença significativa entre as medidas das sépalas das diferentes espécies.")
else:
    print("Não há evidências suficientes para afirmar que há diferença significativa entre as medidas das sépalas das diferentes espécies.")
   
iris_df = pd.DataFrame(data=iris_data.data, columns=iris_data.feature_names)
iris_df['species'] = iris_data.target

# Mapear os números das espécies para seus respectivos nomes
iris_df['species'] = iris_df['species'].map({0: iris_data.target_names[0],
                                             1: iris_data.target_names[1],
                                             2: iris_data.target_names[2]})

# Plotar boxplots para o comprimento das sépalas de cada espécie
plt.figure(figsize=(10, 6))
sns.boxplot(x='species', y='sepal length (cm)', hue='species', data=iris_df)
plt.title('Distribuição do Comprimento das Sépalas por Espécie de Flor Iris')
plt.xlabel('Espécie')
plt.ylabel('Comprimento da Sépala (cm)')
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
plt.show()

# 11. Qual é a relação entre a largura da pétala e a largura da sépala para as diferentes espécies de flores Iris?

# Selecionar apenas as amostras da espécie Iris-setosa
setosa_data = iris_data.data[iris_data.target == 0]

# Calcular a correlação entre o comprimento das pétalas e o comprimento das sépalas para a espécie Iris-setosa
correlation_setosa = np.corrcoef(setosa_data[:, 2], setosa_data[:, 0])[0, 1]

print("Correlação entre comprimento das pétalas e comprimento das sépalas para a espécie Iris-setosa:", correlation_setosa)