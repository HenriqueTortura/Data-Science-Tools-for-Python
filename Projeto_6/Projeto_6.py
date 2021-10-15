import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm

#%% Importando os dados

galton = np.genfromtxt('Galtons_Height_Data.csv', delimiter=',', skip_header=1, usecols=(0,1,2,4,5))

data = np.zeros( (np.size(galton, axis=0), 3) )

data[:,0] = galton[:,-2]
data[:,1] = (galton[:,1] + galton[:,2]) / 2
data[:,2] = galton[:,-1]


#%% Calculando Variâncias e Covariâncias

def Covariance(data, y_ind = 0, x_ind = 1):
    
    y_avg = np.average(data[:,y_ind]) # Y médio (altura média dos filhos) 
    
    x_avg = np.average(data[:,x_ind]) # X médio
    
    # Realiza a Soma
    return np.sum( (data[:,x_ind] - x_avg) * (data[:,y_ind] - y_avg) ) / (np.size(data, axis=0) - 1)

#%% Plot da dispersão

def Plot(data, regression, probabilities, xlabel = 'Altura média dos pais (in)',
         x_ind = 1, label = '1', levels = 10, scatter_color = 'r',
         regression_color = 'black', cmap = cm.plasma):
    
    # O trecho abaixo só garante que o tamanho do gráfico não mude quando adicionando a colorbar
    # Adaptado de https://stackoverflow.com/questions/57530042/matplotlib-let-color-bar-not-affect-size-and-proportions-of-the-plot
    def split_figure_vertical(figsize_1, additional_width, rect_1): 
        """
        figsize_1 is the size of the figure without the color bar
        additional_width is the additional width used for the color bar
        rect_1, rect_2 define where the plotting area and color bar are located
        in their respective sections of the figure
        """
        oldWidth_1 = figsize_1[0]
        
        spacing = additional_width / 30
        
        newWidth = oldWidth_1 + additional_width + spacing
        
        factor_1 = oldWidth_1 / newWidth
        factor_2 = additional_width / newWidth
    
        figsize = (newWidth, figsize_1[1])
    
        fig = plt.figure(figsize=figsize)
    
        rect_1[2] = factor_1
    
        rect_2 = np.zeros(4)
        rect_2[0] += factor_1 + spacing
        rect_2[2] = factor_2
        rect_2[3] = 1
        
        ax1 = fig.add_axes(rect_1)
        ax2 = fig.add_axes(rect_2)
    
        return ax1, ax2
    
    # Plot
    figsize = (10,10)
    rect = [0, 0, 1, 1]
    
    # Plot do mapa de calor das probabilidades
    if probabilities:
        ax1, ax2 = split_figure_vertical(figsize, 1, rect)
        
        # Nomeando variáveis da elipse
        mu_x = np.average(data[:,x_ind])
        mu_y = np.average(data[:,0])
        sigma_x = np.sqrt(Covariance(data, y_ind=x_ind, x_ind=x_ind))
        sigma_y = np.sqrt(Covariance(data, x_ind=0))
        rho = Covariance(data, x_ind = x_ind) / (sigma_x*sigma_y)
        
        # Cria a malha
        y = np.linspace(np.min(data[:,0])-1, np.max(data[:,0])+1, 100)
        x = np.linspace(np.min(data[:,x_ind])-1, np.max(data[:,x_ind])+1, 100)
        xx, yy = np.meshgrid(x, y)
        
        # Cálculo das probabilidades
        exp = ((xx-mu_x)/sigma_x)**2 + ((yy-mu_y)/sigma_y)**2 - 2*rho*((xx-mu_x)/sigma_x)*((yy-mu_y)/sigma_y)
        z = 1/( 2*np.pi*sigma_x*sigma_y*np.sqrt(1-rho**2) ) * np.exp(-1/(2*(1-rho**2)) * exp)

        ax1.contour(xx, yy, z, levels, zorder=2, colors = '#616161', linewidths = 2)
        cp = ax1.contourf(xx, yy, z, levels, cmap = cmap)
        cb = plt.colorbar(cp, cax = ax2)
        cb.ax.tick_params(labelsize=18)
    
    else:
        plt.figure(figsize = figsize)
        ax1 = plt.figure(figsize=figsize).add_axes(rect)
        
    # Plot da dispersão
    ax1.scatter(data[:,x_ind], data[:,0], s=50, zorder=4, c = scatter_color, edgecolors='black')
    
    # Plot da regressão
    if regression[0]:
        x = np.linspace(np.min(data[:,x_ind]), np.max(data[:,x_ind]), 1000)
        y = regression[1]*x + regression[2]
        ax1.plot(x,y, color=regression_color, lw=3, zorder=3,
                 label = 'Regressão: '+r'$\hat{r}(x) = %.3f \cdot x + %.3f  $' % (regression[1], regression[2]))
        ax1.legend(loc='best', fontsize = 20)
    
        
    # Finalização do plot    
    ax1.minorticks_on()
    ax1.grid(axis='both', which='major', linestyle='--', linewidth=1.5)
    ax1.grid(axis='both', which='minor')
    
    ax1.tick_params(labelsize=18)
    ax1.set_xlabel(xlabel, fontsize = 22)
    ax1.set_ylabel('Altura dos filhos (in)', fontsize = 22)
    ax1.set_xlim(np.min(data[:,x_ind])-1, np.max(data[:,x_ind])+1)
    ax1.set_ylim(np.min(data[:,0])-1, np.max(data[:,0])+1)
    
    plt.savefig('Galton_'+label+'.png', dpi=200, bbox_inches='tight')
    
#%% Eixos
def Plot_Eixos(data, regression, probabilities, xlabel = 'Altura média dos pais (in)',
     x_ind = 1, label = '1', levels = 10, scatter_color = 'r',
     regression_color = 'black', cmap = cm.plasma, frame = 5):

    # O trecho abaixo só garante que o tamanho do gráfico não mude quando adicionando a colorbar
    # Adaptado de https://stackoverflow.com/questions/57530042/matplotlib-let-color-bar-not-affect-size-and-proportions-of-the-plot
    def split_figure_vertical(figsize_1, additional_width, rect_1): 
        """
        figsize_1 is the size of the figure without the color bar
        additional_width is the additional width used for the color bar
        rect_1, rect_2 define where the plotting area and color bar are located
        in their respective sections of the figure
        """
        oldWidth_1 = figsize_1[0]
        
        spacing = additional_width / 30
        
        newWidth = oldWidth_1 + additional_width + spacing
        
        factor_1 = oldWidth_1 / newWidth
        factor_2 = additional_width / newWidth
    
        figsize = (newWidth, figsize_1[1])
    
        fig = plt.figure(figsize=figsize)
    
        rect_1[2] = factor_1
    
        rect_2 = np.zeros(4)
        rect_2[0] += factor_1 + spacing
        rect_2[2] = factor_2
        rect_2[3] = 1
        
        ax1 = fig.add_axes(rect_1)
        ax2 = fig.add_axes(rect_2)
    
        return ax1, ax2
    
    # Plot
    figsize = (10,10)
    rect = [0, 0, 1, 1]
    
    # Plot do mapa de calor das probabilidades
    if probabilities:
        ax1, ax2 = split_figure_vertical(figsize, 1, rect)
        
        # Nomeando variáveis da elipse
        mu_x = np.average(data[:,x_ind])
        mu_y = np.average(data[:,0])
        sigma_x = np.sqrt(Covariance(data, y_ind=x_ind, x_ind=x_ind))
        sigma_y = np.sqrt(Covariance(data, x_ind=0))
        rho = Covariance(data, x_ind = x_ind) / (sigma_x*sigma_y)
        
           # Cálculo dos autovalores e autovetores
        delta = np.sqrt((sigma_x**2-sigma_y**2)**2 + 4*(rho*sigma_x*sigma_y)**2)
        lambda_minus = ((sigma_x**2+sigma_y**2) - delta) / 2
        lambda_plus = ((sigma_x**2+sigma_y**2) + delta) / 2
         
        eigenvector_minus = np.array([[(lambda_minus-sigma_y**2)/np.sqrt((lambda_minus-sigma_y**2)**2+(sigma_x*sigma_y*rho)**2)],
                                      [sigma_x*sigma_y*rho/np.sqrt((lambda_minus-sigma_y**2)**2+(sigma_x*sigma_y*rho)**2)]])
        eigenvector_plus = np.array([[(lambda_plus-sigma_y**2)/np.sqrt((lambda_plus-sigma_y**2)**2+(sigma_x*sigma_y*rho)**2)],
                                     [sigma_x*sigma_y*rho/np.sqrt((lambda_plus-sigma_y**2)**2+(sigma_x*sigma_y*rho)**2)]])
    
        # Plot autovetores
        ax1.arrow(mu_x, mu_y, eigenvector_minus[0,0], eigenvector_minus[1,0], width = 0.05,
                  color = 'gray', ec = 'black', zorder=4, label='Autovetores')
        
        ax1.arrow(mu_x, mu_y, eigenvector_plus[0,0], eigenvector_plus[1,0], width = 0.05,
                 color = 'gray', ec = 'black', zorder=4)
        

        # Cria a malha
        y = np.linspace(mu_x - 1.2*frame, mu_x + 1.2*frame, 100)
        x = np.linspace(mu_y - 1.2*frame, mu_y + 1.2*frame, 100)
        xx, yy = np.meshgrid(x, y)
        
        # Cálculo das probabilidades
        exp = ((xx-mu_x)/sigma_x)**2 + ((yy-mu_y)/sigma_y)**2 - 2*rho*((xx-mu_x)/sigma_x)*((yy-mu_y)/sigma_y)
        z = 1/( 2*np.pi*sigma_x*sigma_y*np.sqrt(1-rho**2) ) * np.exp(exp/ (-2*(1-rho**2)))


        ax1.contour(xx, yy, z, levels, zorder=2, colors = '#616161', linewidths = 2)
        cp = ax1.contourf(x, y, z, levels, cmap = cmap)
        cb = plt.colorbar(cp, cax = ax2)
        cb.ax.tick_params(labelsize=18)
    
    else:
        plt.figure(figsize = figsize)
        ax1 = plt.figure(figsize=figsize).add_axes(rect)
        
    # Plot da dispersão
    # ax1.scatter(data[:,x_ind], data[:,0], s=50, zorder=4, c = scatter_color, edgecolors='black')
    
    # Plot da regressão
    if regression[0]:
        x = np.linspace(np.min(data[:,x_ind]), np.max(data[:,x_ind]), 1000)
        y = regression[1]*x + regression[2]
        ax1.plot(x,y, color=regression_color, lw=3, zorder=3,
                 label = 'Regressão: '+r'$\hat{r}(x) = %.3f \cdot x + %.3f  $' % (regression[1], regression[2]))
        ax1.legend(loc='best', fontsize = 20)
    
        
    # Finalização do plot    
    ax1.minorticks_on()
    ax1.grid(axis='both', which='major', linestyle='--', linewidth=1.5)
    ax1.grid(axis='both', which='minor')
    
    ax1.tick_params(labelsize=18)
    ax1.set_xlabel(xlabel, fontsize = 22)
    ax1.set_ylabel('Altura dos filhos (in)', fontsize = 22)

    ax1.set_xlim(mu_x - frame, mu_x + frame)
    ax1.set_ylim(mu_y - frame, mu_y + frame)
    
    plt.savefig('Galton_'+label+'.png', dpi=200, bbox_inches='tight')
    

#%% Regressão linear pelo método dos mínimos quadrados

def Linear_regression(x, y):
    
    beta_1 = np.sum( (x-np.average(x))*(y-np.average(y)) ) / np.sum( (x-np.average(x))**2 )

    beta_0 = np.average(y) - beta_1 * np.average(x)
    
    variance = np.sum((y - beta_1*x - beta_0)**2) / np.size(x)
    
    return beta_1, beta_0, variance

#%% 1. Dispersão sem regressão

regression = np.zeros(4)
Plot(data, regression, False)   


#%% 2. Matriz de variância-covariância
print('Variância da altura dos filhos: ' + str(Covariance(data, x_ind=0)) )

print('\nVariância da altura média dos pais: ' + str(Covariance(data, y_ind=1)) )

print('\nCovariância: ' + str(Covariance(data)) )        

#%% 3. Regressão

regression[0] = 1 # Indica que a regressão deve ser considerada
regression[1:] = np.array(Linear_regression(data[:,1], data[:,0]))
        
Plot(data, regression, False, label = 'regression_1')

#%% Curvas equiprováveis

Plot(data, regression, True, label = 'prob_1', cmap = cm.turbo)

#%% Autovetores

Plot_Eixos(data, regression, True, label = 'eigenvec_1', cmap = cm.turbo, frame=2)


#%% Repetindo para tamanho da família

regression = np.zeros(4)

Plot(data, regression, False, xlabel = 'Tamanho da família',
     x_ind = 2, label = '2') 

print('Variância da altura dos filhos: ' + str(Covariance(data, x_ind=0)) )

print('\nVariância do tamanho da família: ' + str(Covariance(data, y_ind=2, x_ind=2)) )

print('\nCovariância: ' + str(Covariance(data, x_ind=2)) )        

regression[0] = 1 # Indica que a regressão deve ser considerada
regression[1:] = np.array(Linear_regression(data[:,2], data[:,0]))
        
Plot(data, regression, False, xlabel = 'Tamanho da família',
     x_ind = 2, label = 'regression_2')

Plot(data, regression, True, xlabel = 'Tamanho da família',
     x_ind = 2, label = 'prob_2', cmap = cm.turbo)