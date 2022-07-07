import numpy as np

from sklearn.datasets import load_iris

#%%

def ELM(XTrain, YTrain, XTest, Q):
    
    P, N = np.size(XTrain, axis=0), np.size(XTrain, axis=1)
    
    ## Training
    # Normalizing
    X = Standard_Score(XTrain)
    
    # Adding bias
    X = np.concatenate((X, -1*np.ones((1,N))))
    
    # Weights of the hidden layer - of each connection between neurons and inputs
    W = np.random.rand(Q, P+1)
    
    # Hidden layer activation
    Z = g(W @ X)
    
    # Adding bias
    Z = np.concatenate( (Z, -1*np.ones((1,N))) )
    
    # Least squares with Moore-Penrose inverse
    l = 0.001
    M = (YTrain @ Z.T) @ np.linalg.inv(Z @ Z.T + l*(np.identity(Q+1) + Q*np.ones((Q+1, Q+1))))
    
    ## Testing
    N = np.size(XTest, axis=1)
    X = Standard_Score(XTest)
    X = np.concatenate((X, -1*np.ones((1,N))))
    
    Z = g(W @ X)
    Z = np.concatenate( (Z, -1*np.ones((1,N))) )

    Y = M @ Z

    Y = np.argmax(Y, axis = 0)
    Y = Proper_Labels(Y)
    
    return Y


def Standard_Score(X):
    P = np.size(X, axis=0)
    
    mu_X = np.zeros((P,1))
    mu_X[:,0] = np.average(X, axis=1)

    sigma_X = np.zeros((P,1))
    sigma_X[:,0] = np.std(X, axis=1, ddof=1)
    
    return (X - mu_X) / sigma_X
    
def g(x):
    return 1/(1+np.exp(-x))
    
# Change class labels from 0,1,2 to 1 or -1
def Proper_Labels(y_classes):
    Y = np.zeros((np.size(np.unique(y_classes)),
                  np.size(y_classes)))

    for i in range(np.size(np.unique(y_classes))):
        
        Y[i,:] = -1 * np.ones(np.size(y_classes))
        Y[i,np.where(y_classes == np.unique(y_classes)[i])] = 1
    
    return Y

def Select_Train_Data(X, Y, p):
    
    N = np.size(X, axis=1)
    
    indices = np.random.randint(0, N, size = int(N*p))
    
    return X[:,indices], Y[:,indices]


#%%

iris = load_iris()

# Data
X = iris.data.T
Y = Proper_Labels(iris.target)

XTrain, YTrain = Select_Train_Data(X, Y, 0.8)

# Result
result = ELM(XTrain, YTrain, X, 20)

Number_of_misclassifications = np.sum(np.abs(Y - result)) / 4
accuracy = 1 - Number_of_misclassifications / np.size(Y, axis = 1)

print('Accuracy: {}%'.format(accuracy))

#%%
# Code in Matlab - probably wrong

# function [Y] = ELM(XTrain, YTrain, XTest, Q)
    
# [P, N] = np.size(XTrain);

# # Normalizar
# X = zscore(XTrain.T);
# X = X.T;

# X = [X; -ones(1,N)]; # adiciona o bias

# # Pesos da camada oculta
# W = rand(Q, P+1) # Peso de cada conexão entre neurônios com cada entrada

# Ativação da camada oculta
# Z = g(W*X);

# Z = [Z; -ones(1,N+1).T].T # adiciona o bias

# l = 0.001;

# Conexão entre camada intermediária e camda de saída
# M = (YTrain * Z.T) * inverse(Z*Z.T + l*(Identity+Q));

## Test
# XTest = zscore(XTest.T);
# XTest = XTest.T;
# Xtest = [XTest; -ones(1,lenght(XTest))]
    
# Z = g(W*XTest);
# Z = [Z -ones(1,lenght(XTest)).T].T;

# Y = M*Z;

# [~,Y] = max(Y);

# end
    
# function y = g(u)
# y = 1./(1+exp(-u));
# end

