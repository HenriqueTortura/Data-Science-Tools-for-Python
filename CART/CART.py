import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import graphviz


#%% Importando os dados

source = 'https://raw.githubusercontent.com/reisanar/datasets/master/RidingMowers.csv'

data_original = pd.read_csv(source, delimiter= ',')

#%%
def Gini_Total_Impurity(data, covariable):
    
    # Pontos de divisão
    divisions = (( np.unique(data[covariable]) + np.roll(np.unique(data[covariable]), -1) )/2)[:-1]

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

def Grow_Tree(tree, data, current_depth, leaf_signal):
    
    # Todos da mesma classe
    if (np.size(np.unique(data['Ownership'])) == 1 or leaf_signal):
        
        values, counts = np.unique(data['Ownership'], return_counts=True)
        c = values[np.argmax(counts)] # Classe da maioria da folha
        
        node = {
                'id' : np.size(tree),
                'type' : 'leaf',
                'class' : c,
                'samples' : np.size(data, axis=0),
                'depth' : current_depth,
                'data' : data
                }
    
    else:
        possible_node = {}
        
        # Percorre as covariáveis
        for c in data.columns[:-1]:
            # Gera os nós possíveis para cada covariável
            possible_node[c] = Gini_Total_Impurity(data, c)
        
        # Lista as impurezas
        impurities = np.array(list(possible_node.values()))[:,0]
        
        # Escolhe a covariável de menor impureza
        chosen = list(possible_node.keys())[np.where(impurities == np.min(impurities))[0][0]]
        
        # Separa as amostras dado o nó
        branches = [ data[data[chosen] < possible_node[chosen][1]],
                  data[data[chosen] > possible_node[chosen][1]] ]
        
        node = {
                'id' : np.size(tree),
                'type' : 'fork',
                'class' : chosen,
                'samples' : np.size(data, axis=0),
                'impurity' : possible_node[chosen][0],
                'cut' : possible_node[chosen][1],
                'branches' : branches,
                'depth' : current_depth
                }
    
    # Adiciona o nó selecionado à árvore
    tree.append(node)

#%%

def Plant_Tree(depth_limit, data_original):
    # Árvore
    tree = []
    
    # Contador de profundidade e flag para finalizar com folhas
    current_depth = 0
    leaf_signal = False
    
    # Primeiro nó
    Grow_Tree(tree, data_original, current_depth, leaf_signal)
    
            
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
            for data in fork['branches']:
                Grow_Tree(tree, data, current_depth, leaf_signal)
    
    return tree

#%%

depth_limit = 3

tree = Plant_Tree(depth_limit, data_original)

#%%
def Tree_Precision(tree):
    wrong_ones = 0
    
    # Conjunto de folhas
    set_of_leafs = [node for node in tree if node['type'] == 'leaf']
    
    for leaf in set_of_leafs:
    
        data = leaf['data']
        
        values, counts = np.unique(data['Ownership'], return_counts=True)
        c = values[np.argmax(counts)] # Classe da maioria da folha
        
        # Conta os classificados erroneamente
        wrong_ones += np.size(data[data['Ownership'] != c], axis=0)
        
    # Retorna a precisão
    return 1 - wrong_ones/tree[0]['samples']

#%% Função para plotar árvore
def Plot_Tree(tree):

    colors = ['#aae3d7', '#d8e3aa', '#dec7a2']

    g = graphviz.Digraph('g', filename='tree.gv',
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
            id_connection_2 = set_of_nodes[2*i+1]['id']
            
            g.edge(str(id_fork), str(id_connection_1))
            g.edge(str(id_fork), str(id_connection_2))
            
        
    g.view()
    
Plot_Tree(tree)