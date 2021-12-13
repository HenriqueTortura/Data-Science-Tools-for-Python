import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

#%%

# Carregando dados
iris = load_iris()
data = pd.DataFrame(iris.data, columns = iris.feature_names)

# Categorizando os dados em grupos (por espécie)
data['species'] = iris.target_names[iris.target]

print(data.head())


#%% Estudando a minimização de Ĵ

# Percorre cada classe (espécie)
for specie in np.unique(data['species']):
    # Percorre cada atributo    
    for attribute in data.columns[:-1]:
        
        print(specie+' '+attribute[:-5])
        
        # Dados        
        x = data.loc[data['species'] == specie, attribute].values
        
        # Multiplicidade de cada valor
        alphas = np.unique(x, return_counts=True)[1]
        
        k = np.sum((alphas/np.size(x))**2)
        print('{:.4f}'.format(k))


#%%
def cross_validated_risk_estimator(x, m):
    
    # Número de dados
    n = np.size(x)
    
    # Intervalo coberto pelo histograma
    hist_range = np.max(x) - np.min(x)
    
    h = hist_range/m
    
    # Número de dados em cada bin
    nu = np.zeros(m)

    # Percorre cada bin
    for i in range(m):
        
        # Limites inferior e superior de cada bin
        lower_bound = np.min(x) + i*hist_range/m
        upper_bound = np.min(x) + (i+1)*hist_range/m
        
        # Índices dos dados que estão dentro do bin    
        index = np.where(np.logical_and(x >= lower_bound, x < upper_bound))
        
        # Caso especial do último bin de intervalo fechado
        if i == m-1:
            index = np.where(np.logical_and(x >= lower_bound, x <= upper_bound))
        
        # Número de dados em cada bin
        nu[i] = np.size(index)
    
    J = 2/(h*(n-1)) - (n+1)/(h*(n-1)) * np.sum((nu/n)**2)
    
    return J

#%%
def Plot_J(J, m, h, specie, attribute):
    
    colors_a = [(31/255, 119/255, 180/255, 1),
                (255/255, 127/255, 14/255, 1),
                (44/255, 160/255, 44/255, 1),
                (214/255, 39/255, 40/255, 1)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,9))

    ax1.scatter(m, J, s = 80, edgecolors='black',
                facecolor=colors_a[0], linewidths=2, zorder=3)
    ax1.plot(m, J, c=colors_a[0], linewidth=2, zorder=2)
    
    ax2.scatter(h, J, s = 80, edgecolors='black',
                facecolor=colors_a[1], linewidths=2, zorder=3)
    ax2.plot(h, J, c=colors_a[1], linewidth=2, zorder=2)
    
    ax1.set_ylabel(r'$\hat{J}$', fontsize = 22)
    ax1.set_xlabel(r'$m$', fontsize = 22)
    ax2.set_xlabel(r'$h$', fontsize = 22)
    
    ax1.tick_params(axis='both', which='major', labelsize=16)
    ax2.tick_params(axis='both', which='major', labelsize=16)
    ax2.set_yticklabels([])
    
    ax1.grid(axis='both', which='major', linewidth=1.5)
    ax1.grid(axis='both', which='minor', linestyle='--')
    ax2.grid(axis='both', which='major', linewidth=1.5)
    ax2.grid(axis='both', which='minor', linestyle='--')
    
    fig.tight_layout()
    
    plt.savefig(specie+'_'+attribute[:-5]+'_NBC.png', dpi=200, bbox_inches='tight')

    plt.clf()
    plt.close('all')
        
#%%
def Plot_Histo(x, m_best, specie, attribute):
   
    plt.figure(figsize=(16,9))
    
    f = plt.hist(x, m_best, label='Graph', color='#3971cc', edgecolor='#303030',
             linewidth=1.5, zorder=1)
    
    
    plt.xlabel(attribute, fontsize=22)
    plt.ylabel('Contagem', fontsize=22)
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)

    plt.grid(axis='y')
    
    plt.savefig(specie+'_'+attribute[:-5]+'_hist.png', dpi=200, bbox_inches='tight')

    plt.clf()
    plt.close('all')
    
    return f

#%%

# Número de pontos (valores de m)
n_points = 50

J = np.zeros(n_points)
m = np.arange(n_points)+1

# Dicionário das funções f{kj}
f = {}

#Percorre cada classe (espécie)
for specie in np.unique(data['species']):
    
    f[specie] = {}
    
    # Percorre cada atributo    
    for attribute in data.columns[:-1]:
        
        print(specie+' '+attribute[:-5])
        
        # Dados        
        x = data.loc[data['species'] == specie, attribute].values
        
        # Número de dados em cada classe (espécie)
        n_c = np.size(x)
        
        # Intervalo total do histograma
        hist_range = np.max(x) - np.min(x)
        
        # Tamanho de cada bin
        h = hist_range/m

        # Calcula o estimador de validação cruzada do risco
        for i in range(n_points):
            J[i] = cross_validated_risk_estimator(x, m[i])
            
        # Gráfico (função omitida por simplicidade)
        Plot_J(J, m, h, specie, attribute)

        
        # Determina quando o comportamento de J fica linear com m
        tol = 0.0005 # tolerância
        i=0
        while (np.absolute(J[i+2]-J[i+1] - (J[i+1]-J[i])) > tol and (i<n_points-2)):
            i += 1
            
        # Número ótimo de bins
        m_best = np.where(J[:i] == np.min(J[:i]))[0][0] + 1 # +1 por causa do índice
        print(m_best)
        
        # Tamanho ótimo dos bins
        h_best = hist_range/m_best
        
        # Gráfico (função omitida por simplicidade)
        hist = Plot_Histo(x, m_best, specie, attribute)
        
        f[specie][attribute] = np.zeros(3)
        # Estima f_{kj} a partir do histograma
        for i in range(np.size(hist[0])):
            if i == 0:
                f[specie][attribute] = np.array([hist[1][i], hist[1][i+1],
                                                 hist[0][i]/(h_best*n_c)])
            elif i==1:
                f[specie][attribute] = np.stack((f[specie][attribute],
                                                 np.array([hist[1][i], hist[1][i+1],
                                                           hist[0][i]/(h_best*n_c)])))
            else:
                f[specie][attribute] = np.concatenate((f[specie][attribute],
                                                       np.matrix([hist[1][i], hist[1][i+1],
                                                                  hist[0][i]/(h_best*n_c)])), axis=0)


#%% Calculando a probabilidade P(Y=y|X=x)

# Dados
X = np.array(data.values[:,:-1].T, dtype=float)

P = np.zeros((np.size(np.unique(data['species'])), np.size(X, axis=1)))

x = X[:,50]

#%% Classificação

# Percorre dados
for i in range(np.size(X, axis=1)):
# for i in [50, 51, 145]:
    x = X[:,i]
    
    
    f_k = np.ones(np.size(np.unique(data['species'])))

    # Percorre cada classe (espécie)
    for k in range(np.size(np.unique(data['species']))):
        specie = np.unique(data['species'])[k]
        print('\n'+specie)
        
        # Percorre cada atributo    
        for j in range(np.size(data.columns[:-1])):
            attribute = data.columns[:-1][j]
            print(attribute)
            
            # Caso do intervalo do último bin
            if x[j] == f[specie][attribute][-1,1]:
                ind = np.size(f[specie][attribute], axis=0) - 1
                f_kj = f[specie][attribute][ind,2]
                
            else:
                # Índice do bin
                ind = np.where(np.logical_and(x[j] >= f[specie][attribute][:,0],
                                          x[j] < f[specie][attribute][:,1]))[0]
            
                # Probabilidade 0, valor em bin sem observações
                if not ind.size>0:
                    f_kj = 0 
                else:
                    ind = ind[0]
                    f_kj = f[specie][attribute][ind,2]
            
            print(ind)
            
            print(f_kj)
        
            f_k[k] *= f_kj
       
        print(f_k[k])
    
    P[:,i] = f_k / np.sum(f_k)
    
#%%
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
colors_a = [(31/255, 119/255, 180/255, 0.1), (255/255, 127/255, 14/255, 0.1),
            (44/255, 160/255, 44/255, 0.1), (214/255, 39/255, 40/255, 0.1)]

dados = P[:, np.where(P[0,:]==0)[0]][1:, :]

fig, ax = plt.subplots(figsize=(16,16))

lower_lim = -0.05
upper_lim = 1.05

y = np.linspace(lower_lim, upper_lim)

plt.scatter(dados[0,:int(np.size(dados, axis=1)/2)],
            dados[1,:int(np.size(dados, axis=1)/2)], label = 'versicolor',
            linewidths=2, edgecolors='black',
            facecolor=colors[0],
            zorder = 2, s=100)

plt.scatter(dados[0,int(np.size(dados, axis=1)/2):],
            dados[1,int(np.size(dados, axis=1)/2):], label = 'virginica',
            linewidths=2, edgecolors='black',
            facecolor=colors[1],
            zorder = 2, s=100)

plt.plot(y,y, c='black', linewidth=3)

ax.fill_between(y, y, lower_lim, color=colors_a[0])
ax.fill_between(y, y, upper_lim, color=colors_a[1])

plt.grid()
plt.legend(loc='best', fontsize=24)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.xlim(lower_lim, upper_lim)
plt.ylim(lower_lim, upper_lim)
plt.xlabel(r'$P(Y=versicolor | X=x)$', fontsize=24)
plt.ylabel(r'$P(Y=virginica | X=x)$', fontsize=24)
 
plt.savefig('Versicolor_vs_Virginica.png',
            dpi=200, bbox_inches='tight')