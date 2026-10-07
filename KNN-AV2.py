import numpy as np
import time

# KNN - K-Nearest Neighbors (euclidean distance)
#euclidean: calcula a distancia entre dois pontos

# KNN - K-Nearest Neighbors (manhattan distance)
#manhattan: soma o valor absoluto do deslocamento horizontal com o valor absoluto do deslocamento vertical

#KNN - lazy learning
#ele não cria um modelo matemático durante o treino, ele apenas usa os dados de treino para prever os dados de teste.
class KNN:
    def __init__(self, k=5, task="classification", metric="euclidean"):
        self.k = k
        self.task = task
        self.metric = metric # uarda qual distância usar

    #fit so pega os dados e guarda dentro deles mesmos com o self.
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train


    def calculate_distance(self, x1, x2):
        #decide qual cálculo fazer baseado no parâmetro
        if self.metric == "euclidean":
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.metric == "manhattan":
            return np.sum(np.abs(x1 - x2)) #Distância de Manhattan
        else:
            raise ValueError("Métrica deve ser 'euclidean' ou 'manhattan'")

    # recebe uma matriz ou lista contendo vários dados de teste (X_test), basicamente chama o calculate_predict varias vezes mor
    def predict(self, X_test):
        predictions = [self.calculate_predict(x) for x in X_test]
        return np.array(predictions) # retorna array com todas as previsões finais para o conjunto de teste


    # recebe um unico dado de teste
    def calculate_predict(self, x):
        #calcula a distancia de todos os vizinhos
        distances = [self.calculate_distance(x, x_train) for x_train in self.X_train]
        #pega os índices dos K vizinhos mais próximos
        k_indices = np.argsort(distances)[:self.k]
        #descobre quem são esses vizinhos no y_train
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        #faz a predição
        if self.task == "classification": #qual classe apareceu mais vezes entre os vizinhos
            #conta quantas vezes cada classe apareceu
            unique, counts = np.unique(k_nearest_labels, return_counts=True)
            #retorna a classe que apareceu mais vezes
            return unique[np.argmax(counts)]
        elif self.task == "regression": #media dos vizinhos
            return np.mean(k_nearest_labels)


#calcula as metricas
def calcular_metricas(y_test, y_pred):
    vp = np.sum((y_pred == 1) & (y_test == 1)) #verdadeiro positivo
    vn = np.sum((y_pred == 0) & (y_test == 0)) #verdadeiro negativo
    fp = np.sum((y_pred == 1) & (y_test == 0)) #falso positivo
    fn = np.sum((y_pred == 0) & (y_test == 1)) #falso negativo

    #% de acertos gerais (vp + vn) sobre o total
    acuracia = (vp + vn) / (vp + vn + fp + fn)

    #os que o modelo disse ser 1, quantos eram de verdade:
    precisao = vp / (vp + fp) if (vp + fp) > 0 else 0 #if else para nn quebrar no cross validation

    #(sensibilidade) dos 1s (vp) que existiam, quantos o modelo encontrou:
    recall = vp / (vp + fn) if (vp + fn) > 0 else 0 #if else para nn quebrar no cross validation

    #média harmonica (equilíbrio) entre precisão e recall
    f1_score = 2 * (precisao * recall) / (precisao + recall) if (precisao + recall) > 0 else 0  #if else para nn quebrar no cross validation

    return acuracia, precisao, recall, f1_score

def calcular_metricas_regressao(y_test, y_pred, n_atributos):

    y_test = np.array(y_test)
    y_pred = np.array(y_pred)

    # quantas amostras (linhas) de teste
    n_amostras = len(y_test)

    #MAE (Erro Médio Absoluto), a distância absoluta entre a previsão e o valor real
    # np.abs tira o sinal de menos, depois tira a media de todos os erros
    mae = np.mean(np.abs(y_test - y_pred))

    #r²: coeficiente de determinação

    #acham a média real dos dados
    media_real = np.mean(y_test)

    # soma dos erros quadrados = o quanto o modelo errou
    soma_erros_quadrados = np.sum((y_test - y_pred) ** 2)

    # soma da variação total = o quanto os dados variam em relação a  propria media
    soma_variacao_total = np.sum((y_test - media_real) ** 2)

    r2 = 1 - (soma_erros_quadrados / soma_variacao_total)

    #r² ajustado: serve para corrigir o r² quando tem muitos atributos
    # formula = r2 normal, numero amostras (n), número atributos (p)
    # forrmula: 1 - [ (1 - r2) * (n - 1) / (n - p - 1) ]
    r2_ajustado = 1 - ((1 - r2) * (n_amostras - 1) / (n_amostras - n_atributos - 1))

    return mae, r2, r2_ajustado

#validação cruzada:
# avaliar se o modelo generaliza bem: divide o conjunto de dados em K partes iguais, treina com K-1 e testa com 1
# ver se tem overfitting:quando o algoritmo memorizou os dados de treino mas não funciona bem com informações novas
# embaralha os dados, cria K partes iguais, treina com K-1 e testa com 1
def validacao_cruzada(X, y, k_vizinhos=5, metrica="euclidean", k_folds=5):
    # amostras = linhas
    total_amostras = len(X)
    #EMBARALHA OS DADOS
    indices_embaralhados = np.random.permutation(total_amostras)
    #FATIAR OS DADOS EM K FOLDS
    fatias = np.array_split(indices_embaralhados, k_folds)

    notas_acuracia = []
    notas_f1 = []
    tempos_treino = []
    tempos_teste = []
    
    #REPETIR PARA CADA FATIA (RODADAS)
    for i in range(k_folds):
        print("""Rodada %d de %d""" % (i + 1, k_folds))
        #pega a fatia de teste
        indices_teste = fatias[i]
        # cria o bolo de treino (todos menos a fatia de teste)
        indices_treino = []
        for j in range(k_folds):
            if j != i:  # se nao for a fatia de teste
                indices_treino.extend(fatias[j]) # adiciona a fatia de treino

        X_treino = X[indices_treino]
        y_treino = y[indices_treino]

        X_teste = X[indices_teste]
        y_teste = y[indices_teste]

        modelo = KNN(k=k_vizinhos, task="classification", metric=metrica)
        inicio_treino = time.time() #liga
        modelo.fit(X_treino, y_treino)
        fim_treino = time.time() #desliga

        inicio_teste = time.time() #liga
        previsoes = modelo.predict(X_teste)
        fim_teste = time.time() #desliga

        acuracia, precisao, recall, f1_score = calcular_metricas(y_teste, previsoes)

        # append = adicionar elemento no final da lista
        notas_acuracia.append(acuracia)
        notas_f1.append(f1_score)
        tempos_treino.append(fim_treino - inicio_treino)
        tempos_teste.append(fim_teste - inicio_teste)

        print("Acuracia: %.2f" % acuracia)

    # np.mean = média
    # np.std = desvio padrão

    return {
        "acuracia_media": np.mean(notas_acuracia),
        "acuracia_desvio": np.std(notas_acuracia),
        "f1_medio": np.mean(notas_f1),
        "f1_desvio": np.std(notas_f1),
        "tempo_treino_medio": np.mean(tempos_treino),
        "tempo_treino_desvio": np.std(tempos_treino),
        "tempo_teste_medio": np.mean(tempos_teste),
        "tempo_teste_desvio": np.std(tempos_teste)
    }

def validacao_cruzada_regressao(X, y, k_vizinhos=5, metrica="euclidean", k_folds=5):
    total_amostras = len(X)
    indices_embaralhados = np.random.permutation(total_amostras)
    fatias = np.array_split(indices_embaralhados, k_folds)

    notas_mae = []
    notas_r2 = []
    tempos_treino = []
    tempos_teste = []

    # Número de atributos (colunas) para calcular o R2 Ajustado
    # Se X for um vetor 1D, tem 1 atributo. Se for matriz, pega o número de colunas.
    n_atributos = 1 if X.ndim == 1 else X.shape[1]

    for i in range(k_folds):
        indices_teste = fatias[i]
        indices_treino = []
        for j in range(k_folds):
            if j != i:
                indices_treino.extend(fatias[j])

        X_treino, y_treino = X[indices_treino], y[indices_treino]
        X_teste, y_teste = X[indices_teste], y[indices_teste]

        # ATENÇÃO AQUI: task mudou para "regression"
        modelo = KNN(k=k_vizinhos, task="regression", metric=metrica)

        inicio_treino = time.time()
        modelo.fit(X_treino, y_treino)
        fim_treino = time.time()

        inicio_teste = time.time()
        previsoes = modelo.predict(X_teste)
        fim_teste = time.time()

        # ATENÇÃO AQUI: Chama as métricas de regressão
        mae, r2, r2_ajustado = calcular_metricas_regressao(y_teste, previsoes, n_atributos)

        notas_mae.append(mae)
        notas_r2.append(r2)
        tempos_treino.append(fim_treino - inicio_treino)
        tempos_teste.append(fim_teste - inicio_teste)

    return {
        "mae_medio": np.mean(notas_mae),
        "mae_desvio": np.std(notas_mae),
        "r2_medio": np.mean(notas_r2),
        "r2_desvio": np.std(notas_r2),
        "tempo_treino_medio": np.mean(tempos_treino),
        "tempo_treino_desvio": np.std(tempos_treino),
        "tempo_teste_medio": np.mean(tempos_teste),
        "tempo_teste_desvio": np.std(tempos_teste)
    }

# Dados de treino
X_train = np.array([[3, 2], [4, 5], [1, 1], [2, 2], [5, 4]])
y_train = np.array([1, 1, 0, 0, 1])

# Dados de teste inventados
X_test = np.array([[3, 3], [1, 2], [5, 5]])
y_test = np.array([1, 0, 1]) # O gabarito real

print("euclidiana")
modelo_euclidiano = KNN(k=3, task="classification", metric="euclidean")
modelo_euclidiano.fit(X_train, y_train)
previsoes_euclidianas = modelo_euclidiano.predict(X_test)

acuracia_euclidiana, precisao_euclidiana, recall_euclidiana, f1_euclidiana = calcular_metricas(y_test, previsoes_euclidianas)

print("Previsões: ", previsoes_euclidianas)
print("Acurácia: ", acuracia_euclidiana)
print("F1-Score: ", f1_euclidiana)


print("manhattan")
modelo_manhattan = KNN(k=3, task="classification", metric="manhattan")
modelo_manhattan.fit(X_train, y_train)
previsoes_manhattan = modelo_manhattan.predict(X_test)

acuracia_manhattan, precisao_manhattan, recall_manhattan, f1_manhattan = calcular_metricas(y_test, previsoes_manhattan)

print("Previsões: ", previsoes_manhattan)
print("Acurácia: ", acuracia_manhattan)
print("F1-Score: ", f1_manhattan)