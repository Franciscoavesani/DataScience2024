#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 22:16:00 2024

@author: alisson
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from scipy.stats import (norm, gamma, expon, lognorm, beta, t, logistic, gumbel_r, gumbel_l, genextreme,
                         uniform, pareto, pearson3, ks_2samp)
import kalepy as kale
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from mlxtend.plotting import plot_decision_regions
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate


#Inicialização do Banco de Dados


iris = load_iris()
df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])
print(iris.target_names)

#Análise de Dados:

#1. Qual é a média, mediana e desvio padrão do comprimento das pétalas de todas as espécies?
targets = df['target'].unique()
for target in targets:
    print(df.loc[df['target'] == target].describe())

#6. Usando o KDE, como é a distribuição da largura das pétalas para cada espécie de flor?
def getProbabilityDensity():
    return {
        "Normal"             : norm,
        "Gamma"              : gamma,
        "Exponential"        : expon,
        "Lognormal"          : lognorm,
        "Beta"               : beta,
        "Student-t"          : t,
        "Logistic"           : logistic,
        "Gumbel Right"       : gumbel_r,
        "Gumbel Left"        : gumbel_l,
        "Uniform"            : uniform,
        "Generalized Pareto" : pareto,
        "Pearson Type III"   : pearson3
    }

def calculateProbabilityDensity(df):
    ks_columns = {}
    ks_results = {}
    pdfs = getProbabilityDensity()
    for column in df.columns:
        for idx, (name, pdf) in enumerate(pdfs.items()):
            if pdf is None:
                ks_results[name] = (np.nan, np.nan)
                continue
            data = df[column]
            params = pdf.fit(data)
            
            rv = pdf(*params)
            ks_stat, p_value = ks_2samp(data, rv.rvs(size=len(data)))
            ks_results[name] = (ks_stat, p_value)
        ks_columns[column] = ks_results
    return ks_columns
    
        
ks_results = calculateProbabilityDensity(df)
plt.tight_layout()
# Print KS Test results
for column in df.columns:
    if(column == 'target'):
        continue
    print(column)
    print("\n Kolmogorov-Smirnov (KS) test results:\n")
    print("{:<25} {:<20} {:<20}".format("Probability distribution", "KS statistic", "p-value"))
    print("-" * 65)
    for pdf_name, (ks_stat, p_value) in ks_results[column].items():
        print("{:<25} {:<20.5f} {:<20.5f}".format(pdf_name, ks_stat, p_value))
    print("\n")
plt.show()


#2. Qual espécie de flor tem a maior largura média de sépala? E a menor?
print(iris.target_names[0])

#3. Existe uma correlação entre o comprimento e a largura das pétalas? E entre o comprimento das pétalas e o comprimento das sépalas?
correlation = df[df.columns[0:-1]].corr()
sns.heatmap(correlation, annot = True, fmt=".2f", linewidths=.6)
plt.show()

#4. Usando o algoritmo de agrupamento K-means, quantos clusters são ideais para representar as diferentes espécies de flores Iris?
def kmeans_alisson(df,x1,x2):

    X = df[[df.columns[x1],df.columns[x2]]]
    Y = df[df.columns[-1]]
    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size = 0.7, random_state = 42)
    y_train = y_train.values.astype(np.int_)
    
    # Redução de dimensionalidade com PCA
    #pca = PCA(n_components=2)
    #X_pca = pca.fit_transform(x_train)
    nkf = 5
    # Lista para armazenar a variabilidade explicada
    inertia = []
    vscore = []
    vk = []
    print(x_train)
    # Testar diferentes números de clusters
    for n_clusters in range(1, 11):
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(x_train) 
        inertia.append(kmeans.inertia_)
        
        model = KNeighborsClassifier(n_neighbors=n_clusters, metric = 'euclidean')
        model.fit(x_train, y_train)
        # Plotting decision regions
        
        
        
        
        plt.figure()
        plot_decision_regions(x_train.values, y_train, clf=model, legend=4)
        plt.xlabel(df.columns[x1])
        plt.ylabel(df.columns[x2])
        plt.title('Decision Regions: n_clusters = '+str(n_clusters))
        #plt.savefig('knn_' + str(k)+'.eps')
        plt.show()
        
        cv = cross_validate(model, x_train.values, y_train, cv=nkf)
        #print('k:', k, 'accurace:', cv['test_score'].mean())
        vscore.append(cv['test_score'].mean()) 
        vk.append(n_clusters)

        
    fig, ax = plt.subplots()
    ax.plot(vk, vscore, marker='o', linestyle='--', label='Validação')
    ax.set_ylabel('Pontuação de Validação')
    ax2 = ax.twinx()
    ax2.plot(vk, inertia, marker='s', linestyle='-', color='orange', label='Inércia')
    ax2.set_ylabel('Inércia')
    ax.set_xlabel('Número de Clusters (K)')
    
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    plt.grid(True)
    plt.show()


kmeans_alisson(df,0,1)
kmeans_alisson(df,2,3)
#5. Quais características (comprimento da sépala, largura da sépala, comprimento da pétala, largura da pétala) são mais úteis para distinguir entre as diferentes espécies de flores?
#largura da sépala e +1 qualquer.


#7. Existem outliers nos dados? Como eles afetam a distribuição das espécies de flores? Plote os dados em scatter plots e dê destaque com cor vermelha para os outliers.
plt.figure(dpi=300)
sns.pairplot(df, hue="target")
plt.show()


#8. Qual é a probabilidade de encontrar uma flor Iris-versicolor com largura de pétala entre 1.5 e 2.0 cm?
pAB = len((df.loc[(df['target'] == 1) &
           (df['petal width (cm)'] > 1.5) &
           (df['petal width (cm)'] < 2 )])
          )/len(df.loc[(df['target'] == 1)])
 
#9. Como as medidas das sépalas se comparam entre as espécies de flores? Existe uma diferença significativa?
plt.figure(dpi=300)
sns.scatterplot(x= df['sepal length (cm)'],y=df['sepal width (cm)'], hue=df['target'])
plt.show()


#10. Existe uma relação entre o comprimento das pétalas e o comprimento das sépalas em uma determinada espécie de flor?
plt.figure(dpi=300)
sns.scatterplot(x= df['sepal length (cm)'],y=df['petal length (cm)'], hue=df['target'])
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.show()

#11. Qual é a relação entre a largura da pétala e a largura da sépala para as diferentes espécies de flores Iris?
plt.figure(dpi=300)
sns.scatterplot(x= df['sepal width (cm)'],y=df['petal width (cm)'], hue=df['target'])
plt.xlim(0, 5)
plt.ylim(0, 5)
plt.show()