import numpy as np
import time

# COMO RODAR, no Terminal do Powershell
# python -m venv env
# env\Scripts\activate

# python NOMEARQUIVO.py


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
        # propagacao pelas camadas ocultas
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

# //////////////////////////////////////////////////////// parte pedro!!!!


dados = np.genfromtxt('file16a868cf35f5.csv', delimiter=',', skip_header=1)


def entradar(d):
    X = np.column_stack(
        (d[:, 1], d[:, 2], d[:, 3], d[:, 4], d[:, 5], d[:, 6], d[:, 7], d[:, 8], d[:, 9], d[:, 10], d[:, 11],
         d[:, 12], d[:, 13], d[:, 14], d[:, 15], d[:, 16], d[:, 17], d[:, 18], d[:, 19], d[:, 20], d[:, 21], d[:, 22], d[:, 23]))
    y = d[:, 25]
    return [X, y]


def fatiar(dados, camadas):

    metricas = []

    dados = np.random.permutation(dados)  # embarallhado

    # separei a ultima fatia pra fazer o predict
    entradadaoutrafatiapraypred = entradar(dados[1581:])[0]
    y_teste = entradar(dados[1581:])[1]
    # as primeiras 4 fatias sao pro treinamento
    entrada, y = entradar(dados[:1581])

    tempoantes = time.time()
    modelo = MLP(23, camadas)
    modelo.fit(entrada, y)
    metricas.append(time.time()-tempoantes)

    tempoantes = time.time()
    y_pre = modelo.predict(entradadaoutrafatiapraypred)
    metricas.append(time.time()-tempoantes)

    metricas.append(r2_score(y_teste, y_pre))
    metricas.append(MAE(y_teste, y_pre))
    metricas.append(RMSE(y_teste, y_pre))
    metricas.append(MSE(y_teste, y_pre))

    return metricas


def r2_score(y_true, y_pred):
    return 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)


def MAE(y_true, y_pred):
    return (1/(y_pred.shape[0]) * np.sum(np.abs(y_true - y_pred)))


def RMSE(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def MSE(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def memetricar(camadas):

    memetricas = []

    for i in range(5):
        memetricas.append(np.array(fatiar(dados, camadas)))

    memetricas = np.array(memetricas)

    print("media tempo de treino:", np.mean(memetricas[:, 0]))
    print("media tempo de predicao:", np.mean(memetricas[:, 1]))
    print("media R2:", np.mean(memetricas[:, 2]))
    print("media MAE:", np.mean(memetricas[:, 3]))
    print("rmse: ", np.mean(memetricas[:, 4]))
    print("mse: ", np.mean(memetricas[:, 5]))
    print("media R2 ajustado:", 1-((1-np.mean(memetricas[:, 2]))*(527-1)/(527-23-1)))


memetricar([5, 5, 5])
memetricar([9, 9, 9])
memetricar([5, 5, 5, 5, 5, 5])
memetricar([9, 9, 9, 9, 9, 9])
