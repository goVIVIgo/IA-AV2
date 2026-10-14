import numpy as np
import time

dados = np.genfromtxt('file16a868cf35f5.csv', delimiter=',', skip_header=1)

# tres das colunas sao NaN, e nao tem como saber o que cada uma delas é ou representa, entao terei que escolher arbitrariamente a predita e preditoras
# nan 1 -> 23 nan 25 nan


class MultipleRegression:  # construtor
    def __init__(self, x, y):
        self.X = x  # entrada
        self.y = y  # saida

    def fit(self):
        return np.linalg.pinv(self.X.T @ self.X) @ self.X.T @ self.y

    def predict(self, x_new, beta):
        return x_new @ beta  # aqui o x_new ja vem com a coluna de ones


def entradar(d):
    X = np.column_stack(
        (np.ones(len(d[:, 0])), d[:, 1], d[:, 2], d[:, 3], d[:, 4], d[:, 5], d[:, 6], d[:, 7], d[:, 8], d[:, 9], d[:, 10], d[:, 11],
         d[:, 12], d[:, 13], d[:, 14], d[:, 15], d[:, 16], d[:, 17], d[:, 18], d[:, 19], d[:, 20], d[:, 21], d[:, 22], d[:, 23]))
    y = d[:, 25]
    return [X, y]


def fatiar(dados):

    metricas = []

    dados = np.random.permutation(dados)  # embarallhado

    # separei a ultima fatia pra fazer o predict
    entradadaoutrafatiapraypred = entradar(dados[:-527])[0]
    # as primeiras 4 fatias sao pro treinamento
    entrada, y = entradar(dados[:1581])

    tempoantes = time.time()
    modelo = MultipleRegression(entrada, y)
    beta = modelo.fit()
    metricas.append(time.time()-tempoantes)

    tempoantes = time.time()
    y_pre = modelo.predict(entradadaoutrafatiapraypred, beta)
    metricas.append(time.time()-tempoantes)

    metricas.append(r2_score(y, y_pre))
    metricas.append(MAE(y, y_pre))

    return metricas


def r2_score(y_true, y_pred):
    return 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)


def MAE(y_true, y_pred):
    return (1/(y_pred.shape[0]) * np.sum(np.abs(y_true - y_pred)))


memetricas = []

for i in range(5):
    memetricas.append(np.array(fatiar(dados)))

memetricas = np.array(memetricas)

print("media tempo de treino:", np.mean(memetricas[:, 0]))
print("media tempo de predicao:", np.mean(memetricas[:, 1]))
print("media R2:", np.mean(memetricas[:, 2]))
print("media MAE:", np.mean(memetricas[:, 3]))
