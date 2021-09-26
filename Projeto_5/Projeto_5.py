import numpy as np
import matplotlib.pyplot as plt

import matplotlib.animation as animation

import time

#%%
def Plot_Rho(mu_x, mu_y, sigma_x_values, sigma_y_values, rho_values, k, label, frame=3):
    
    # Definindo pontos da elipse
    x_lim = [-frame, frame]
    y_lim = [-frame, frame]
    x = -np.linspace(x_lim[0], x_lim[1], 1000)
    y = np.linspace(y_lim[0], y_lim[1], 1000)
    X,Y = np.meshgrid(x,y)
    
    
    # Plot
    fig, ax = plt.subplots(figsize=(16,16))
    # plt.tight_layout()
    fig.subplots_adjust(left=0.08, bottom=0.08, right=0.98, top=0.98)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    def animate(i):
        ax.clear()
        
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_xlabel(r'$x$', fontsize=26)
        ax.set_ylabel(r'$y$', fontsize=26)
        ax.grid()
        ax.tick_params(axis='both', which='major', labelsize=22)
        
        sigma_x = sigma_x_values[i]
        sigma_y = sigma_y_values[i]
        rho = rho_values[i]
        
        # Cálculo dos autovalores e autovetores
        delta = np.sqrt((sigma_x**2-sigma_y**2)**2 + 4*(rho*sigma_x*sigma_y)**2)
        lambda_minus = ((sigma_x**2+sigma_y**2) - delta) / 2
        lambda_plus = ((sigma_x**2+sigma_y**2) + delta) / 2
        
        eigenvector_minus = np.array([[(lambda_minus-sigma_y**2)/np.sqrt((lambda_minus-sigma_y**2)**2+(sigma_x*sigma_y*rho)**2)],
                                     [sigma_x*sigma_y*rho/np.sqrt((lambda_minus-sigma_y**2)**2+(sigma_x*sigma_y*rho)**2)]])
        eigenvector_plus = np.array([[(lambda_plus-sigma_y**2)/np.sqrt((lambda_plus-sigma_y**2)**2+(sigma_x*sigma_y*rho)**2)],
                                     [sigma_x*sigma_y*rho/np.sqrt((lambda_plus-sigma_y**2)**2+(sigma_x*sigma_y*rho)**2)]])

            # eigenvector_minus = np.array([[sigma_x*sigma_y*rho/np.sqrt((lambda_minus-sigma_x)**2+(sigma_x*sigma_y*rho)**2)],
            #                               [(lambda_minus-sigma_y)/np.sqrt((lambda_minus-sigma_x)**2+(sigma_x*sigma_y*rho)**2)]])
            # eigenvector_plus = np.array([[(lambda_plus-sigma_y)/np.sqrt((lambda_plus-sigma_y)**2+(sigma_x*sigma_y*rho)**2)],
            #                              [sigma_x*sigma_y*rho/np.sqrt((lambda_plus-sigma_x)**2+(sigma_x*sigma_y*rho)**2)]])

        # Equação da elipse
        eqn = ((X-mu_x)/sigma_x)**2 + ((Y-mu_y)/sigma_y)**2 - 2*rho*((X-mu_x)/sigma_x)*((Y-mu_y)/sigma_y)
        A = -2* (1-rho**2)* np.log( 2*np.pi* k* sigma_x*sigma_y*np.sqrt(1-rho**2) )
        
        # Plot elipse
        ax.contour(X,Y,eqn,[A], cmap='winter', linewidths=3)
    
        # Plot autovetores
        ax.text(0.2*eigenvector_minus[0,0], eigenvector_minus[1,0]/2, r'$ê_-$',
                 size = 26, verticalalignment='top', horizontalalignment='right',
                 color = colors[3], bbox={'facecolor': 'white', 'alpha': 0.8,
                                        'pad': 0.1, 'boxstyle': 'round'})
        ax.arrow(mu_x, mu_y, eigenvector_minus[0,0], eigenvector_minus[1,0], width = 0.05,
                  color = colors[3], ec = 'black')
        
        ax.text(1.1*eigenvector_plus[0,0], eigenvector_plus[1,0]/2, r'$ê_+$',
                 size = 26, verticalalignment='top', horizontalalignment='right',
                 color=colors[3], bbox={'facecolor': 'white', 'alpha': 0.8,
                                        'pad': 0.1, 'boxstyle': 'round'})
        ax.arrow(mu_x, mu_y, eigenvector_plus[0,0], eigenvector_plus[1,0], width = 0.05,
                 color = colors[3], ec = 'black')
        
        # Plot valores
        text = '\n'.join((
                # r'$\mu_x=%.2f$' % (mu_x, ),
                # r'$\mu_y=%.2f$' % (mu_x, ),
                r'$\sigma_x=%.2f$' % (sigma_x, ),
                r'$\sigma_y=%.2f$' % (sigma_y, ),
                r'$\rho=%.2f$' % (rho, ),))
        ax.text(-0.9*frame, 0.9*frame, text,
                 size = 28, verticalalignment='top', horizontalalignment='left',
                 color='#3971cc', bbox={'facecolor': 'white', 'alpha': 0.8,
                                        'pad': 0.5, 'boxstyle': 'round'})
        
    # plt.rcParams['animation.ffmpeg_path'] = '/home/lordemomo/anaconda3/bin/ffmpeg'
    
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=int(np.size(rho_values)/15), metadata=dict(artist='Me'))
    ani = animation.FuncAnimation(fig, animate, frames=np.size(rho_values),
                                             repeat=True)
    ani.save(label+'_'+str(np.size(rho_values))+'.mp4', writer=writer)
    # plt.show(ani) 
    
#%% Implementação da normal multivariada
def Multivariate_Normal(p, x, sigma, mu):
    x = x - mu
    exp = np.matmul(x.T, np.linalg.inv(sigma))
    exp = np.matmul(exp, x)
    return 1/((2*np.pi)**(p/2) * (np.sqrt(np.linalg.det(sigma)))) * np.exp(-0.5*exp)[0,0]

#%% Função para controlar caso usuário rode sem olhar o código
def yes_or_no(question): #Straight from https://stackoverflow.com/questions/47735267/while-loop-with-yes-no-input-python
    reply = str(input(question+' (y/n): ')).lower().strip()
    if reply[0] == 'y':
        return 1
    elif reply[0] == 'n':
        return 0
    else:
        return yes_or_no("Please Enter")

#%% 1. Cálculo dos autovalores e autovetores

p = 2 # Bivariada

mu_x = 0
mu_y = 0

k = 0.01


#%% Variando rho
if yes_or_no('\n Variar $\rho$ ?'):
    
    rho_values = np.linspace(0.01, 0.99, num=100)
    sigma_x_values = np.ones(np.size(rho_values)) * 1.5
    sigma_y_values = np.ones(np.size(rho_values)) * 1.25
    
    start = time.time()
    
    Plot_Rho(mu_x, mu_y, sigma_x_values, sigma_y_values, rho_values, k, 'Rho', frame=4.5)
    
    end =  time.time() 
    
    print(end-start)

#%% Variando sigma_x
if yes_or_no('\n Variar $\sigma_x$ ?'):

    sigma_x_values = np.linspace(0.5, 5, num=100)
    rho_values = np.ones(np.size(sigma_x_values))*0.5
    sigma_y_values = np.ones(np.size(sigma_x_values))* 1
    
    start = time.time()
    
    Plot_Rho(mu_x, mu_y, sigma_x_values, sigma_y_values, rho_values, k, 'Sigma_x', frame=8)
    
    end =  time.time()
    
    print(end-start)

#%% Variando sigma_y

if yes_or_no('\n Variar $\sigma_y$ ?'):
    
    sigma_y_values = np.linspace(0.5, 5, num=100)
    rho_values = np.ones(np.size(sigma_y_values))*0.5
    sigma_x_values = np.ones(np.size(sigma_y_values))* 1
    
    start = time.time()
    
    Plot_Rho(mu_x, mu_y, sigma_x_values, sigma_y_values, rho_values, k, 'Sigma_y', frame=8)

    end =  time.time()

    print(end-start)