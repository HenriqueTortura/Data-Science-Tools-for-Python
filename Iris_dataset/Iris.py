import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Carrega todos os dados
iris = load_iris()
# Separa rótulos de cada subconjunto de dados por conveniência
labels = [feature.capitalize()[:-4] for feature in iris['feature_names']]

#%% Boxplots
plt.figure(figsize=(16,9))

plt.boxplot(iris["data"], labels = labels)

plt.grid()
plt.xticks(fontsize = 18)
plt.yticks(fontsize = 14)
plt.ylabel("cm", fontsize = 18)
plt.savefig('box_plot.png', dpi=200, bbox_inches='tight')

#%% Cumulative Distribution Functions
def Plot_Empirical_Cumulative_Distribution_Function(data, label):

    # Obtém um vetor dos valores únicos (x) e outro com a quantidade em que cada valor aparece (counts)
    x, counts = np.unique(data, return_counts=True)
    # Garante primeiro degrau do valor incial
    x = np.concatenate(([x[0]], x))
    
    # Gera a ECDF a partir do vetor de contagem partindo do zero
    ecdf = np.cumsum(np.concatenate(([0], counts)))
    ecdf = ecdf/ecdf[-1] # Normalização da ECDF
    
    # Plot
    fig = plt.figure(figsize=(16,9))
    ax = fig.add_axes([0, 0, 1, 1])
    
    plt.plot(x, ecdf, drawstyle='steps-post', linewidth = 4)
    
    ax.minorticks_on()
    plt.grid(axis='both', which='major', linestyle='--', color='#878787', linewidth=1.5)
    plt.grid(axis='x', which='minor', color='#adadad')
    plt.xticks(fontsize = 18)
    plt.yticks(ticks=np.linspace(0,1,11) ,fontsize = 18)
    plt.xlabel('x (cm)', fontsize = 22)
    plt.ylabel('F(x)', fontsize = 22)
    plt.ylim(0,1)
    plt.savefig(label+'ECDF.png', dpi=200, bbox_inches='tight')
    
# Percorrendo para os diferentes subconjuntos de dados
for i in range(len(iris['feature_names'])):
    Plot_Empirical_Cumulative_Distribution_Function(iris.data[:,i], labels[i])