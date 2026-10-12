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


#calcula as metricas da classificação
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

    residuos = y_test - y_pred

    #MAE (erro médio absoluto) =distância absoluta entre a previsão e o valor real
    # np.abs tira o sinal de menos, depois tira a media de todos os erros
    mae = np.mean(np.abs(residuos))

    #r²: coeficiente de determinação (começo dos calculos)
    #acha a média real dos dados
    media_real = np.mean(y_test)

    # soma dos erros quadrados = o quanto o modelo errou
    soma_erros_quadrados = np.sum(residuos ** 2)

    # soma da variação total = o quanto os dados variam em relação a  propria media
    soma_variacao_total = np.sum((y_test - media_real) ** 2)

    #r²: coeficiente de determinação (juntando tudo)
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
def validacao_cruzada(X, y, k_vizinhos=5, metrica="euclidean", k_folds=5, task="classification"):
    # amostras = linhas
    total_amostras = len(X)

    #permutation = EMBARALHA OS DADOS
    indices_embaralhados = np.random.permutation(total_amostras)

    # array_split = FATIAR OS DADOS EM K FOLDS
    fatias = np.array_split(indices_embaralhados, k_folds)

    tempos_treino = []
    tempos_teste = []

    # se for classificação guarda as notas de classificação e se for regressão guarda as notas de regressão
    if task == "classification":
        notas_acuracia = []
        notas_f1 = []
    elif task == "regression":
        notas_mae = []
        notas_r2 = []
        # pega a quantidade de colunas (atributos) para o cálculo do R2 Ajustado
        n_atributos = 1 if X.ndim == 1 else X.shape[1]

    #ele roda k(5) vezes e pra cada vez ele escolhe um bolinho para ser amostra de teste(1/5) e o restante para treino(4/5 = k-1)
    for i in range(k_folds):
        print("Rodada", i + 1, "de", k_folds)

        #pega a fatia de teste
        indices_teste = fatias[i]

        # cria o bolinho de treino (k-1 = todos menos a fatia de teste)
        indices_treino = []

        for j in range(k_folds):
            if j != i:  # se nao for a fatia de teste
                indices_treino.extend(fatias[j]) # adiciona a fatia de treino

        X_treino = X[indices_treino]
        y_treino = y[indices_treino]
        # X = matriz de atributos (caracteristicas), y = vetor de classes (gabaritos)
        X_teste = X[indices_teste]
        y_teste = y[indices_teste]

        # passa a task(classificação ou regressão) e a metrica(euclidean ou manhattan) para o modelo knn, que ele cria a cada iteração
        modelo = KNN(k=k_vizinhos, task=task, metric=metrica)

        inicio_treino = time.time() #liga
        modelo.fit(X_treino, y_treino) #fit armazena os dados das k-1 fatias que cobraram
        fim_treino = time.time() #desliga

        inicio_teste = time.time()
        previsoes = modelo.predict(X_teste) #passa a fatia que sobrou para ver se ele sabe oq ele ta fazendo msm
        fim_teste = time.time()

        # append = adicionar a uma lista
        tempos_treino.append(fim_treino - inicio_treino)
        tempos_teste.append(fim_teste - inicio_teste)

        #metricas
        if task == "classification":
            # calcula as metricas da classificação (acuracia, precision, recall, f1)
            acuracia, precisao, recall, f1 = calcular_metricas(y_teste, previsoes) #compara as previsoes com as respostas reais
            notas_acuracia.append(acuracia)
            notas_f1.append(f1)
            print("Acuracia da rodada:", acuracia)

        elif task == "regression":
            # calcula as metricas da regressão (MAE, R2, R2 ajustado)
            mae, r2, r2_ajustado = calcular_metricas_regressao(y_teste, previsoes, n_atributos) #compara as previsoes com as respostas reais
            notas_mae.append(mae)
            notas_r2.append(r2)
            print("MAE da rodada:", mae)

    # np.mean tira a média
    # np.std tira o desvio padrão

    # tempo = oq eles tem em comim
    resultados = {
        "tempo_treino_medio": np.mean(tempos_treino),
        "tempo_treino_desvio": np.std(tempos_treino),
        "tempo_teste_medio": np.mean(tempos_teste),
        "tempo_teste_desvio": np.std(tempos_teste)
    }

    #update = atualizar no dicionário
    if task == "classification":
        resultados.update({
            "acuracia_media": np.mean(notas_acuracia),
            "acuracia_desvio": np.std(notas_acuracia),
            "f1_medio": np.mean(notas_f1),
            "f1_desvio": np.std(notas_f1)
        })
    elif task == "regression":
        resultados.update({
            "mae_medio": np.mean(notas_mae),
            "mae_desvio": np.std(notas_mae),
            "r2_medio": np.mean(notas_r2),
            "r2_desvio": np.std(notas_r2)
        })

    return resultados

print("KNN - classificação (COMEÇO)")
dados_classificacao = np.genfromtxt(r"C:\Users\vivif\OneDrive\Desktop\TrabalhosSillys\SEXTO SEMESTRE\IA\AULA 02\trabalho AV2\gina_prior.csv", delimiter=",")
print("dados usados para classificação",dados_classificacao)

#pega todas as colunas exceto a ultima (x = todos os atributos, y = classe)
X_classificacao = dados_classificacao[:, :-1]
#pega apenas a ultima coluna (y = classe)
y_classificacao = dados_classificacao[:, -1]

y_classificacao[y_classificacao == -1] = 0

print("KNN - classificação")
print("linhas", X_classificacao.shape[0]) #shape[0] = retorna os elementos da matriz, nesse caso colunas
print("colunas", X_classificacao.shape[1]) #shape = retorna os elementos da matriz, nesse caso linhas

#hiperparametros
k = 5
folds = 5

print("euclidiana")
resultados_euclidiana_classificacao = validacao_cruzada(X_classificacao, y_classificacao, k_vizinhos=k, k_folds=folds, task="classification", metrica="euclidean")
#acuracia media de 95%
print("Acuracia media: ", resultados_euclidiana_classificacao["acuracia_media"])
#f1 medio de 95%
print("F1 media: ", resultados_euclidiana_classificacao["f1_medio"])
#treino do knn nn faz merda nenhuma ele so guarda os dados, ai da pouco tempo
print("Tempo de treino medio: ", resultados_euclidiana_classificacao["tempo_treino_medio"])
#teste do knn demora quanto maior for pq ele vai fazendo os calculos das distancias com os vizinhos, TODOS os vizinhos
print("Tempo de teste medio: ", resultados_euclidiana_classificacao["tempo_teste_medio"])

print("manhattan")
resultados_manhattan_classificacao = validacao_cruzada(X_classificacao, y_classificacao, k_vizinhos=k, k_folds=folds, task="classification", metrica="manhattan")
#acuracia media de 93%
print("Acuracia media: ", resultados_manhattan_classificacao["acuracia_media"])
#f1 medio de 93%
print("F1 media: ", resultados_manhattan_classificacao["f1_medio"])
##treino do knn nn faz merda nenhuma ele so guarda os dados, ai da pouco tempo (foi so um segundo)
print("Tempo de treino medio: ", resultados_manhattan_classificacao["tempo_treino_medio"])
#teste do knn demora quanto maior for pq ele vai fazendo os calculos das distancias com os vizinhos, TODOS os vizinhos (demorou 6 e pouco)
print("Tempo de teste medio: ", resultados_manhattan_classificacao["tempo_teste_medio"])

#o knn da clasificação foi mto bem pq ele estava analisando pixels, que são super proximos(vizinhos), então foi mto facil pra ele prever e fazer o voto majoritario
print("KNN - classificação (FIM)")

print("KNN - regressão(COMEEÇO)")

dados_regressao = np.genfromtxt(r"C:\Users\vivif\OneDrive\Desktop\TrabalhosSillys\SEXTO SEMESTRE\IA\AULA 02\trabalho AV2\file16a868cf35f5.csv", delimiter=",", )
print("dados usados para regressão",dados_regressao)

print("KNN - regressão")
#pula a primeira coluna e ignora as 3 ultimas
X_regressao = dados_regressao[:, 1:-3]
#pega apenas a penultima coluna pq o tempo de execução ta la
y_regressao = dados_regressao[:, -2]

print("linhas", X_regressao.shape[0]) #shape[0] = retorna os elementos da matriz, nesse caso colunas
print("colunas", X_regressao.shape[1]) #shape = retorna os elementos da matriz, nesse caso linhas

print("euclidiana")
resultados_euclidiana_regressao = validacao_cruzada(X_regressao, y_regressao, k_vizinhos=k, k_folds=folds, task="regression", metrica="euclidean")
#erra o erro medio por 1066 segundos = 17min, baixo
print("MAE media: ", resultados_euclidiana_regressao["mae_medio"])
#vai de 0 a 1, aqui foi 0.14 ent ele ta so chutando as coisa a partir de determinado momento
print("R2 media: ", resultados_euclidiana_regressao["r2_medio"])
print("Tempo de treino medio: ", resultados_euclidiana_regressao["tempo_treino_medio"])
print("Tempo de teste medio: ", resultados_euclidiana_regressao["tempo_teste_medio"])

print("manhattan")
resultados_manhattan_regressao = validacao_cruzada(X_regressao, y_regressao, k_vizinhos=k, k_folds=folds, task="regression", metrica="manhattan")
#erra o erro medio por 1074 segundos = 18min, baixo
print("MAE media: ", resultados_manhattan_regressao["mae_medio"])
#vai de 0 a 1, aqui foi 0.15 ent ele ta so chutando as coisa a partir de determinado momento
print("R2 media: ", resultados_manhattan_regressao["r2_medio"])
print("Tempo de treino medio: ", resultados_manhattan_regressao["tempo_treino_medio"])
print("Tempo de teste medio: ", resultados_manhattan_regressao["tempo_teste_medio"])

#o knn da regressão teve dificuldade com a previsão dos dados pq os dados poderiam estar muito longe e atrapalhar ele na previsão
print("KNN - regressão (FIM)")

