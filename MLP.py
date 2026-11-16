import numpy as np

def sigmoid(x): 
    return 1 / (1 + np.exp(-np.clip(x, -50, 50)))
def tanh(x):
    return np.tanh(x)
def relu(x):
    return np.maximum(0, x)
def ativacao(x, nome): 
    if nome == "sigmoid":
        return sigmoid(x)
    if nome == "relu":
        return relu(x)
    return tanh(x)
def derivada(x, nome):
    if nome == "sigmoid":
        s = sigmoid(x)
        return s * (1 - s)
    if nome == "relu":
        return (x > 0).astype(float)
    return 1 - np.tanh(x) ** 2

class MLP:
    def __init__(self, entradas, camadas, taxa=0.001, epocas=300, funcao="tanh"): 
        self.taxa = taxa
        self.epocas = epocas
        self.funcao = funcao        
        # define a  estrutura da rede (entradas -camadas ocultas - saida)
        tamanhos = [entradas] + camadas + [1]
        self.pesos = []
        self.bias = []        
        # inicializa pesos e bias aleatoriamentee
        
        for i in range(len(tamanhos) - 1):
            peso = np.random.randn(tamanhos[i], tamanhos[i + 1]) * 0.1
            b = np.zeros((1, tamanhos[i + 1]))
            self.pesos.append(peso)
            self.bias.append(b)

    def forward(self, X): 
        saidas = [X]
        somas = []
        atual = X        
        #propagacao pelas camadas ocultas
        for i in range(len(self.pesos) - 1):
            z = atual @ self.pesos[i] + self.bias[i]
            atual = ativacao(z, self.funcao)
            somas.append(z)
            saidas.append(atual)            
        # camada de saida (linear)
        saida = atual @ self.pesos[-1] + self.bias[-1]
        somas.append(saida)
        saidas.append(saida)
        return saidas, somas

    def fit(self, X, y):
        y = y.reshape(-1, 1)        
        # loops de treinamento pr epoca
        for _ in range(self.epocas):
            saidas, somas = self.forward(X)            
            erro = saidas[-1] - y
            delta = (2 / len(X)) * erro            
            grad_pesos = [None] * len(self.pesos)
            grad_bias = [None] * len(self.bias)            
            # gradientes da camada final 
            grad_pesos[-1] = saidas[-2].T @ delta
            grad_bias[-1] = np.sum(delta, axis=0, keepdims=True)            
            # backpropagation nass camadas ocultas
            for i in range(len(self.pesos) - 2, -1, -1):
                delta = (
                    delta @ self.pesos[i + 1].T
                ) * derivada(somas[i], self.funcao)
                grad_pesos[i] = saidas[i].T @ delta
                grad_bias[i] = np.sum(delta, axis=0, keepdims=True)                
            # atualizacao dos pesos e bias com gradiente descendente
            
            for i in range(len(self.pesos)):
                self.pesos[i] -= self.taxa * grad_pesos[i]
                self.bias[i] -= self.taxa * grad_bias[i]

    def predict(self, X):
        saidas, _ = self.forward(X)
        return saidas[-1].ravel()