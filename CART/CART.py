import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

import graphviz # Biblioteca para gerar árvore de classificação


#%% Importando os dados

source = 'https://raw.githubusercontent.com/reisanar/datasets/master/RidingMowers.csv'

data_original = pd.read_csv(source, delimiter= ',')

#%%
def Gini_Total_Impurity(data, covariable, t):
    # Parâmetros
    #### data: DataFrame (pandas) do conjunto total de dados sobre o qual a
    #### impureza total de Gini será avaliada
    ####
    #### covariable: covariável (string, coluna de DataFrame) sobre a qual a
    #### impureza total de Gini será avaliada 
    ####
    #### t: ponto de corte
    
    # Retorna
    #### I: Impureza total de Gini
        
    # Conjuntos
    A_1 = data[data[covariable]<t]
    A_2 = data[data[covariable]>t]
    
    # Frações das observações
    p_1 = {}
    p_2 = {}
    # Percorre classes
    for c in np.unique(data['Ownership']):
        # If resolve problema pelo conjunto de treinamento ser comparável ao de validação
        if np.size(A_1, axis=0) == 0:
            p_1[c] = 0
        else:
            p_1[c] = np.size(A_1[A_1['Ownership']==c], axis=0) / np.size(A_1, axis=0)
        
        if np.size(A_2, axis=0) == 0:
            p_2[c] = 0
        else:
            p_2[c] = np.size(A_2[A_2['Ownership']==c], axis=0) / np.size(A_2, axis=0)
    
    # Impurezas/Índices de Gini   
    gamma_1 = 1 - np.sum( np.array(list(p_1.values()))**2 )
    gamma_2 = 1 - np.sum( np.array(list(p_2.values()))**2 )
    
    # Impureza Total de Gini
    I = ( np.size(A_1, axis=0) * gamma_1 + np.size(A_2, axis=0) * gamma_2 ) / np.size(data, axis=0)

    return I


#%%
def Minimal_Gini_Total_Impurity(data, covariable):
    # Parâmetros
    #### data: DataFrame (pandas) do conjunto total de dados sobre o qual a
    #### impureza total de Gini será avaliada
    ####
    #### covariable: covariável (string, coluna de DataFrame) sobre a qual a
    #### impureza total de Gini será avaliada 
    
    # Retorna
    #### I_min: Impureza total de Gini mínima
    ####
    #### t_best: Ponto de corte da covariável que minimiza a impureza total de 
    #### Gini
    
    # Pontos de divisão
    divisions = (( np.unique(data[covariable]) + np.roll(np.unique(data[covariable]), -1) )/2)[:-1]
    
    # Se não houverem pontos de divisão
    if divisions.size == 0:
        return 1, 0
        
    I = np.zeros(np.size(divisions))
    
    # Percorre cada ponto de divisão
    for i in range(np.size(divisions)):
        t = divisions[i]
        
        # Conjuntos
        A_1 = data[data[covariable]<t]
        A_2 = data[data[covariable]>t]
        
        # Frações das observações
        p_1 = {}
        p_2 = {}
        # Percorre classes
        for c in np.unique(data['Ownership']):
            p_1[c] = np.size(A_1[A_1['Ownership']==c], axis=0) / np.size(A_1, axis=0)
            p_2[c] = np.size(A_2[A_2['Ownership']==c], axis=0) / np.size(A_2, axis=0)
        
        # Impurezas/Índices de Gini   
        gamma_1 = 1 - np.sum( np.array(list(p_1.values()))**2 )
        gamma_2 = 1 - np.sum( np.array(list(p_2.values()))**2 )
        
        # Impureza Total de Gini
        I[i] = ( np.size(A_1, axis=0) * gamma_1 + np.size(A_2, axis=0) * gamma_2 ) / np.size(data, axis=0)

    # Impureza mínima
    I_min = np.min(I)
    
    # Ponto de divisão que minimiza a impureza    
    t_best = divisions[np.where(I == np.min(I))][0]

    return I_min, t_best

#%%

def Grow_Tree(tree, data, current_depth, leaf_signal, fork_region):
    # Função que adiciona um novo nó (ramificação ou folha) à árvore.
    
    # Parâmetros
    #### tree: árvore (lista de dicionários), que carrega os nós (dicionários
    #### com informações pertinentes).
    ####
    #### data: conjunto de dados (DataFrame) sobre os quais deve-se criar um
    #### novo nó.
    ####
    #### current_depth: nível em que o novo nó se insere.
    ####
    #### leaf_signal: sinaliza que o novo deve ser uma folha por estar no
    #### nível máximo permitido.
    
    # Todos da mesma classe ou profundidade máxima
    if (np.size(np.unique(data['Ownership'])) == 1 or leaf_signal):
        
        values, counts = np.unique(data['Ownership'], return_counts=True)
        c = values[np.argmax(counts)] # Classe da maioria da folha
        
        node = {
                'id' : np.size(tree),
                'type' : 'leaf',
                'class' : c,
                'samples' : np.size(data, axis=0),
                'depth' : current_depth,
                'data' : data,
                'region' : fork_region.copy()
                }
    
    else:
        possible_node = {}
        
        # Percorre as covariáveis
        for c in data.columns[:-1]:
            # Gera os nós possíveis para cada covariável
            possible_node[c] = Minimal_Gini_Total_Impurity(data, c)
        
        # Lista as impurezas
        impurities = np.array(list(possible_node.values()))[:,0]
        
        # Escolhe a covariável de menor impureza
        chosen = list(possible_node.keys())[np.where(impurities == np.min(impurities))[0][0]]
        
        # Separa as amostras dado o nó
        branches = [ data[data[chosen] < possible_node[chosen][1]],
                     data[data[chosen] > possible_node[chosen][1]] ]
        
        # Definindo regiões
        if chosen == 'Income':
            region_1 = fork_region.copy()
            region_1[1] = possible_node[chosen][1]
            
            region_2 = fork_region.copy()
            region_2[0] = possible_node[chosen][1]
            
        else:
            region_1 = fork_region.copy()
            region_1[3] = possible_node[chosen][1]
            
            region_2 = fork_region.copy()
            region_2[2] = possible_node[chosen][1]
        
        node = {
                'id' : np.size(tree),
                'type' : 'fork',
                'class' : chosen,
                'samples' : np.size(data, axis=0),
                'impurity' : possible_node[chosen][0],
                'cut' : possible_node[chosen][1],
                'branches' : branches,
                'depth' : current_depth,
                'regions' : [region_1, region_2]
                }
    
    # Adiciona o nó selecionado à árvore
    tree.append(node)

#%%

def Plant_Tree(depth_limit, data_original):
    # Função que cria a árvore de classificação até uma profundidade máxima.
    
    # Parâmetros
    #### depth_limit: profunidade (número de níveis) máxima.
    #### data_original: conjunto de dados sobre os quais se gerará a árvore.
    
    # Retorna
    #### tree: lista de dicionários, na qual cada dicionário é um nó
    #### (ramificação ou folha).
    
    # Árvore
    tree = []
    
    # Contador de profundidade e flag para finalizar com folhas
    current_depth = 0
    leaf_signal = False
    
    # Região inicial (2x região dos dados)
    mu_x = ( np.min(data_original['Income']) + np.max(data_original['Income']) ) / 2
    mu_y = ( np.min(data_original['Lot_Size']) + np.max(data_original['Lot_Size']) ) / 2
    a = np.array([np.min(data_original['Income']), np.max(data_original['Income']),
                  np.min(data_original['Lot_Size']), np.max(data_original['Lot_Size'])])
    region_0 = 2*a - np.array([mu_x, mu_x, mu_y, mu_y])
    
    # Primeiro nó
    Grow_Tree(tree, data_original, current_depth, leaf_signal, region_0)
            
    for current_depth in range(1, depth_limit+1):
        
        # Conjunto das ramificações
        set_of_forks = [node for node in tree if (node['depth'] == current_depth-1 and node['type'] == 'fork')]
    
        # Se não hoverem ramificações, árvore acaba
        if not set_of_forks:
            break
        
        # Se chegarmos na última profundidade permitida, teremos apenas folhas
        if current_depth == depth_limit:
            leaf_signal = True
            
        # Árvore continua a crescer nas ramificações
        for fork in set_of_forks:
            for i in range(len(fork['branches'])):
                Grow_Tree(tree, fork['branches'][i], current_depth,
                          leaf_signal, fork['regions'][i])
    
    return tree


#%%
def Tree_Precision(tree):
    wrong_ones = 0
    
    # Conjunto de folhas
    set_of_leafs = [node for node in tree if node['type'] == 'leaf']
    
    for leaf in set_of_leafs:
        
        if leaf['samples'] > 0:
    
            data = leaf['data']
            
            values, counts = np.unique(data['Ownership'], return_counts=True)
            c = values[np.argmax(counts)] # Classe da maioria da folha
            
            # Conta os classificados erroneamente
            wrong_ones += np.size(data[data['Ownership'] != c], axis=0)
        
    # Retorna a precisão
    return 1 - wrong_ones/tree[0]['samples']

#%%

def Use_Tree(tree, data_original):
    # Função que classifica um conjunto de dados usando uma árvore pré-definida.
    
    # Parâmetros
    #### tree: lista de dicionários, na qual cada dicionário é um nó, para
    #### classificar dados
    #### data_original: conjunto de dados que serão classificados.
    
    # Retorna
    #### new_tree: lista de dicionários, na qual cada dicionário é um nó
    #### (ramificação ou folha).
    
    # Árvore nova
    new_tree = []
    for i in range(len(tree)):
        new_tree.append(dict(tree[i]))
        
    # Primeiro nó
    t = tree[0]['cut']
    covariable = tree[0]['class']
    branches = [ data_original[data_original[covariable] < t],
                 data_original[data_original[covariable] > t] ]
    
    new_tree[0]['impurity'] = Gini_Total_Impurity(data_original, covariable, t)
    new_tree[0]['branches'] = branches
    new_tree[0]['samples'] = np.size(data_original, axis=0)
    
    # Profundidade máxima da árvore
    max_depth = tree[-1]['depth']
    
    # Percorre outras profundidades
    for d in range(1, max_depth+1):
        
        set_of_previous_forks = [node for node in new_tree if (node['type'] == 'fork' and node['depth'] == d-1)]
        set_of_nodes = [node for node in tree if node['depth'] == d]
        
    
        # Percorre ramificações anteriores
        for i in range(len(set_of_previous_forks)):
            fork = set_of_previous_forks[i]
            
            # Percorre cada caminho da bifurcação
            for j in range(len(fork['branches'])):
                data = fork['branches'][j]
                node = set_of_nodes[2*i+j]
                index = node['id']

                # Atualiza ramificação
                if node['type'] == 'fork':
                    t = node['cut']
                    covariable = node['class']
                    branches = [ data[data[covariable] < t],
                                 data[data[covariable] > t] ]
                    
                    new_tree[index]['impurity'] = Gini_Total_Impurity(data, covariable, t)
                    new_tree[index]['branches'] = branches
                    new_tree[index]['samples'] = np.size(data, axis=0)
        
                # Atualiza folha
                if node['type'] == 'leaf':
                    new_tree[index]['data'] = data
                    new_tree[index]['samples'] = np.size(data, axis=0)
    
    return new_tree
            
        
#%% Função para plotar árvore
def Plot_Tree(tree, name=''):

    colors = ['#aae3d7', '#d8e3aa', '#dec7a2']

    g = graphviz.Digraph('g', filename='tree'+name+'.gv',
                         format = 'png',
                         node_attr={'shape': 'record', 'height': '.1'})
    
    
    # Criando os nós
    for i in range(np.size(tree)):
        
        if tree[i]['type'] == 'leaf':
            g.attr('node', shape='box')
            
            text = '\n'.join((
                            'Folha',
                            'Amostras: {}'.format(tree[i]['samples']),
                            'Classe: {}'.format(tree[i]['class'])
                            ))
            
            if tree[i]['class'] == 'Owner':
                color = colors[0]
            else:
                color = colors[1]
            
        else:
            g.attr('node', shape='ellipse')
            
            text = '\n'.join((
                            'Ramificação',
                            'Amostras: {}'.format(tree[i]['samples']),
                            '{} <= {}'.format(tree[i]['class'], tree[i]['cut']),
                            'Impureza de Gini Total: {:.3f}'.format(tree[i]['impurity'])
                            ))
            
            color = colors[2]
            
        g.node(str(i), label = text, style='filled', fillcolor = color)
    
    
    set_of_forks = [node for node in tree if node['type'] == 'fork']
    
    # Percorre níveis de profundidade que tem ramificações 
    for d in range(np.max([fork['depth'] for fork in set_of_forks])+1):
        
        # Conjuntos de ramificações do nível
        set_of_forks_in_d = [node for node in tree if (node['type'] == 'fork' and node['depth'] == d)]
        
        for i in range(np.size(set_of_forks_in_d)):
            # Identificação da ramificação
            id_fork = set_of_forks_in_d[i]['id']
            
            set_of_nodes = [node for node in tree if node['depth'] == d+1]
            
            id_connection_1 = set_of_nodes[2*i]['id']
            n_1 = set_of_nodes[2*i]['samples']
            id_connection_2 = set_of_nodes[2*i+1]['id']
            n_2 = set_of_nodes[2*i+1]['samples']
            
            g.edge(str(id_fork), str(id_connection_1), label=str(n_1))
            g.edge(str(id_fork), str(id_connection_2), label=str(n_2))
            
        
    g.view()
    
#%%

def Plot_Phase_Space(tree, data_original, name=''):

    fig, ax = plt.subplots(figsize=(16,9))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    colors_a = [(31/255, 119/255, 180/255, 0.2),
                (255/255, 127/255, 14/255, 0.2),
                (44/255, 160/255, 44/255, 0.2),
                (214/255, 39/255, 40/255, 0.2)]
    
    plt.scatter(data_original[data_original['Ownership']=='Owner']['Income'],
                data_original[data_original['Ownership']=='Owner']['Lot_Size'],
                label = 'Proprietários',
                c = colors[0], edgecolors='black', linewidths=2,
                zorder = 2, s = 100)
    
    plt.scatter(data_original[data_original['Ownership']=='Nonowner']['Income'],
                data_original[data_original['Ownership']=='Nonowner']['Lot_Size'],
                label = 'Não Proprietários',
                c = colors[1], edgecolors='black', linewidths=2,
                zorder = 2, s = 100)
        
    x_lower = 30
    x_upper = 114
    
    y_lower = 13.5
    y_upper = 24
    
    set_of_forks = [node for node in tree if node['type'] == 'fork']
    
    for fork in set_of_forks:
        if fork['class'] == 'Income':
            plt.vlines(fork['cut'],
                        ymin = fork['regions'][0][2],
                        ymax = fork['regions'][0][3],
                        linewidth=3,
                        linestyle = '--', color = 'black')
            
        else:
            plt.hlines(fork['cut'],
                        xmin = fork['regions'][0][0],
                        xmax = fork['regions'][0][1],
                        linewidth=3,
                        linestyle = '--', color = 'black')
            
    set_of_leafs = [node for node in tree if node['type'] == 'leaf']
    for leaf in set_of_leafs:
        xv = [leaf['region'][0], leaf['region'][1],
              leaf['region'][1], leaf['region'][0]]
        yv = [leaf['region'][2], leaf['region'][2],
              leaf['region'][3], leaf['region'][3]]
        
        if leaf['class'] == 'Owner':
            color = colors_a[0]
        else:
            color = colors_a[1]
        
        plt.fill(xv, yv, color = color)
    
    plt.legend(loc='upper left', fontsize = 22)
    
    plt.ylabel('Lot Size', fontsize = 24)
    plt.xlabel('Income', fontsize = 24)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    
    plt.xlim((x_lower, x_upper))
    plt.ylim((y_lower, y_upper))
    
    ax.minorticks_on()
    
    # ax.xaxis.set_major_locator(MultipleLocator(1))
    # ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    # ax.yaxis.set_minor_locator(MultipleLocator(0.025))
    
    ax.grid(axis='both', which='major', linewidth=1.5)
    ax.grid(axis='both', which='minor', linestyle='--')
    
    plt.savefig('phase_space'+name+'.png', dpi=200, bbox_inches='tight')
    
    
#%%

max_depth = 6 # Maior profundidade testada
n_t = 16 # Tamanho do conjunto de treinamento

n_r = 100 # Número de realizações para cada profundidade

# Precisão do conjunto de treinamento e de validação
p_t = np.zeros(n_r)
p_v = np.zeros(n_r)

precision_t = np.zeros((max_depth, 2))
precision_v = np.zeros((max_depth, 2))

# Varrendo profundidades
for i in range(1, max_depth+1):
    # print(i)
    
    # Varrendo amostragens
    for j in range(n_r):
        print('{}: {} / {}'.format(i, j, n_r))
        
        # Conjunto de treinamento
        data_t = data_original.sample(n=n_t)
        
        # Conjunto de validação
        data_v = data_original.iloc[np.delete(np.array(data_original.index),
                                              np.array(data_t.index))]
        
        # Árvore criada pelo conjunto de treinamento
        tree = Plant_Tree(i, data_t)
        
        # Aplicando a árvore ao conjunto de validação
        tree_v = Use_Tree(tree, data_v)
        
        p_t[j] = Tree_Precision(tree)
        p_v[j] = Tree_Precision(tree_v)
    
    # Precisões tomadas pela média das realizações
    precision_t[i-1,:] = [np.average(p_t), np.std(p_t, ddof = 1)]
    precision_v[i-1,:] = [np.average(p_v), np.std(p_v, ddof = 1)]
    
#%%
def Plot_Cross_Validation(precision_t, precision_v, max_depth, n_r):
    fig, ax = plt.subplots(figsize=(16,9))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    plt.errorbar(range(1, max_depth+1), precision_t[:,0],
                 yerr = precision_t[:,1]/np.sqrt(n_r),
                 fmt='o-', linewidth = 3, c = colors[0],
                 label = 'Conjunto de Treinamento')
    
    plt.errorbar(range(1, max_depth+1), precision_v[:,0],
                 yerr = precision_v[:,1]/np.sqrt(n_r),
                 fmt='o-', linewidth = 3, c = colors[1],
                 label = 'Conjunto de Validação')
        
    plt.legend(loc='best', fontsize = 22)
    
    plt.ylabel('Precisão', fontsize = 24)
    plt.xlabel('Profundidade', fontsize = 24)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    
    ax.minorticks_on()
    
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.025))
    
    ax.grid(axis='both', which='major', linewidth=1.5)
    ax.grid(axis='both', which='minor', linestyle='--')
    
    plt.savefig('cross_validation.png', dpi=200, bbox_inches='tight')
    
Plot_Cross_Validation(precision_t, precision_v, max_depth, n_r)

#%%

# Conjunto de treinamento
data_t = data_original.sample(n=16)

# Árvore criada pelo conjunto de treinamento
tree = Plant_Tree(4, data_t)

# Aplicando a árvore ao conjunto de validação
tree_v = Use_Tree(tree, data_original)
#%%
Plot_Tree(tree_v, name='v')

print(Tree_Precision(tree_v))

Plot_Phase_Space(tree, data_original, name='v')

#%% Árvore de profundidade 4 gerada com o conjunto de todos os dados
 
tree = Plant_Tree(4, data_original)

Plot_Tree(tree)

print(Tree_Precision(tree))

Plot_Phase_Space(tree, data_original)