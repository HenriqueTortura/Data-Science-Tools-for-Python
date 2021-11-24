import numpy as np
import scipy.stats as stats
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

#%%
# Soma das diferenças ao quadardo intra-grupo
def Calculate_SS_err(data, column):
    
    SS_err = 0
    
    for specie in np.unique(data['species']):
        Y_i_bar = np.average(data.loc[data['species'] == specie, column].values)
        
        for Y_ij in data.loc[data['species'] == specie, column].values:
            SS_err += (Y_ij - Y_i_bar)**2
            
    return SS_err


# Soma das diferenças entre médias ao quadrado inter-grupos
#(com peso do número de variáveis)
def Calculate_SS_trat(data, column):
    
    SS_trat = 0
    Y_bar = np.average(data[column])
    
    for specie in np.unique(data['species']):
        n_i = np.size(data.loc[data['species'] == specie, column].values)
        Y_i_bar = np.average(data.loc[data['species'] == specie, column].values)
        
        SS_trat += n_i*(Y_i_bar - Y_bar)**2
        
    return SS_trat


#%%

# Criando a tabela de saída
anova = pd.DataFrame(columns = [i.replace(' (cm)', '') for i in iris.feature_names],
                     index=['SS_err', 'SS_trat', 'SS_tot', 'F', 'p-value'])


# Realizando a ANOVA para cada coluna
for column in data.columns[0:-1]:
    anova_column = column.replace(' (cm)', '')
    
    # Número de pontos
    N = np.size(data[column])
    
    # Número de grupos 
    g = np.size(np.unique(data['species']))
    
    anova.at['SS_err', anova_column] = Calculate_SS_err(data, column)
    
    anova.at['SS_trat', anova_column] = Calculate_SS_trat(data, column)
    
    anova.at['SS_tot', anova_column] = anova[anova_column]['SS_trat'] + anova[anova_column]['SS_err'] 
    
    anova.at['F', anova_column] = ((N-g) * anova[anova_column]['SS_trat']) / ((g-1) * anova[anova_column]['SS_err'])
    
    
    anova.at['p-value', anova_column] = stats.f.sf(anova[anova_column]['F'],
                                                   g-1, N-g)
   
    
#%%
print(anova.columns)

for i in range(np.size(anova['petal length'])):
    if i % 2:
        color = ' \\rowcolor{LightCyan}'
    else: color = ''
    
    if i < np.size(anova['petal length'])-2:
        print('${}'.format(anova.index[i].replace('_', '_{'))+'}$'+' & {:.6f} & {:.6f} & {:.6f} & {:.6f} \\\ \hline {}'.format(anova['sepal length'][i],
                                                          anova['sepal width'][i],
                                                          anova['petal length'][i],
                                                          anova['petal width'][i],
                                                          color))
    elif i < np.size(anova['petal length'])-1:
        print('${}$'.format(anova.index[i].replace('_', '_{'))+' & {:.6f} & {:.6f} & {:.6f} & {:.6f} \\\ \hline {}'.format(anova['sepal length'][i],
                                                          anova['sepal width'][i],
                                                          anova['petal length'][i],
                                                          anova['petal width'][i],
                                                          color))
        
    else:
        print('${}$'.format(anova.index[i].replace('_', '_{'))+' & {:.6e} & {:.6e} & {:.6e} & {:.6e} \\\ \hline {}'.format(anova['sepal length'][i],
                                                          anova['sepal width'][i],
                                                          anova['petal length'][i],
                                                          anova['petal width'][i],
                                                          color))


#%%
    
fig, ax = plt.subplots(figsize=(16,9))
# plt.title(Title+', T='+str(T), fontsize=10)

colors = ['#cc7400', '#ccb800', '#85cc00', '#00b096']

x = np.linspace(0, 1200, 10**6)
ax.plot(x, stats.f.pdf(x, g-1, N-g), color='blue', lw=3)

plt.axvline(stats.f.isf(0.05, g-1, N-g), color='red', linewidth=2)
plt.text(0.05+25, stats.f.sf(0.05, g-1, N-g),
         r'$\alpha = 0.05$',
         size = 16, verticalalignment='top', horizontalalignment='left',
         color='red',
         bbox={'facecolor': 'white', 'alpha': 0.7,
               'pad': 0.5, 'boxstyle': 'round'})

f_values = np.sort(anova.loc['F'].values)
for i in range(np.size(f_values)):
    plt.axvline(f_values[i], color=colors[i], linewidth=3)
    if i<2:
        plt.text(f_values[i]+25, stats.f.sf(f_values[i], g-1, N-g),
                anova.columns.values[np.where(anova.iloc[-2,:] == f_values[i])[0][0]],
                size = 16, verticalalignment='top', horizontalalignment='left',
                color=colors[i],
                bbox={'facecolor': 'white', 'alpha': 0.7,
                      'pad': 0.5, 'boxstyle': 'round'})
    else:
        plt.text(f_values[i]-25, (10**5)*stats.f.sf(f_values[i], g-1, N-g),
                anova.columns.values[np.where(anova.iloc[-2,:] == f_values[i])[0][0]],
                size = 16, verticalalignment='bottom',
                horizontalalignment='right',
                color=colors[i],
                bbox={'facecolor': 'white', 'alpha': 0.7,
                      'pad': 0.5, 'boxstyle': 'round'})

ax.set_yscale('log')

plt.ylabel(r'$f_{\; F-S}^{\; g-1,N-g}(F)$', fontsize = 22)
plt.xlabel(r'$F$', fontsize = 22)

plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)


ax.grid(axis='both', which='major', linewidth=1.5)
ax.grid(axis='both', which='minor', linestyle='--')

plt.savefig('test.png', dpi=200, bbox_inches='tight')


#%% Tukey test

# Valor de q (mesmo para todos os casos) 
q = stats.studentized_range.ppf(0.95, g, N-g)

# Realizando o teste de Tukey para cada coluna
for column in data.columns[0:-1]:
    
    MS = Calculate_SS_err(data, column) / (N-g)
    n = N/g # Caso de n_1 = ... = n_g
    
    # Calculando a distância significativa honesta
    HSD = q*np.sqrt(MS/n)
    
    # Obtendo a média para cada grupo
    Y_i_bar = {}
    for specie in np.unique(data['species']):
        Y_i_bar[specie] = np.average(data.loc[data['species'] == specie, column].values)

    # Ordena em ordem crescente as médias
    Y_i_bar = dict(sorted(Y_i_bar.items(), key=lambda item: item[1]))
    
    # Vetor de valores ordenados
    values = np.array(list(Y_i_bar.values()))
    
    print('{} & {:.6f} & {:.6f} & {:.6f} \\\ \hline'.format(column, HSD,
                                                       values[1]-values[0],
                                                       values[2]-values[1] ))