import numpy as np
import time

#NAIVE BAYES - UNIVARIADO - CLASSIFICAÇÃO
#ele assume que os dados sequem a distribuição normal (a curva do sino la)
#muito naive: faz suposições ingenuas de que os dados são independentes uns dos outros
#eager learner (ansioso) - ele cria o modelo matemático durante o treino (fit), por isso eh cheio de calculo bem ali embaixo

#treino (fit): faz 3 coisas pra CADA classe
#1) calcula a media de cada coluna (atributo) da classe
#2) calcula a variancia de cada coluna (atributo) da classe
#3) calcula a prior (probabilidade a priori) de cada classe

#test (predict): usa o modelo matemático criado no fit pra fazer a previsão (tipo um concurso de contest pokemon com a melhor pontuação), pra CADA amostra de X_test
#1) usa a pdf (probability density function) pra calcular a probabilidade de cada classe e ve se oq ta nela pertece aquela curva
#2) soma de logaritmos para evitar overflow que poderia vir ao multiplicar
#3) escolhe a classe com maior probabilidade (somando o a chance dos pixels de cada classe + a chance da classe existir)
# esse predict eh bem rapido pq so joga as coisa na pdf e soma os logaritmos

#a desvantagem do naive bayes eh que se os dados forem mto dependentes uns dos outros essa naivedade pode ser prejudicial

class NaiveBayesUnivariado:
    # o modelo matematico do naive bayes é criado no fit, DIFERENTE DO KNN que NAO FAZ ISSO NO FIT
    def fit(self, X_train, y_train):

        # usa o shape pra descobrir quantas linhas e quantas colunas tem o X_train
        n_amostras, n_atributos = X_train.shape

        #descobre quais classses tem no y_train(gabarito)
        self.classes = np.unique(y_train)
        n_classes = len(self.classes)

        #np.zeros cria uma matriz de zeros com as dimensões indicadas
        # media = é a media ne porr
        self.media = np.zeros((n_classes, n_atributos), dtype=np.float64)
        #variancia = desvio padrão ao quadrado
        self.variancia = np.zeros((n_classes, n_atributos), dtype=np.float64)
        #prior = probabilidade total de cada classe
        self.prior = np.zeros(n_classes, dtype=np.float64)

        # calcula a media, a variancia e o prior para CADA classe
        for idx, classe in enumerate(self.classes):

            # filtra o x_train para pegar apenas as amostras da classe naquela linha
            X_classe = X_train[y_train == classe]

            # calcula a media de cada coluna (atributo) da classe
            self.media[idx, :] = X_classe.mean(axis=0)

            # calcula a variancia de cada coluna (atributo) da classe
            # + 1e-9 evita divisão por zero
            self.variancia[idx, :] = X_classe.var(axis=0) + 1e-9

            # prior calcula a probabilidade total da classe (quantidade de amostras avaliadas da classe / total de amostras)
            self.prior[idx] = X_classe.shape[0] / float(n_amostras)


    # recebe várias linhas de teste e faz uma previsão para cada uma
    def predict(self, X_test):
        # oara cada linha 'x' no teste chama a função de calcula probabilidades
        previsoes = [self.calcular_probabilidade(x) for x in X_test]
        return np.array(previsoes) # retorna array com todas as previsões finais

    # função para calcular a probabilidade UMA linha
    def calcular_probabilidade(self, x):
        posteriors = [] # guarda as chances finais de cada classe em uma lista

        # para cada classe
        for idx, classe in enumerate(self.classes):

            # pega a prior e aplica logaritmo, o log transform as multiplicações em somas para tornar o calculo numericamente mais estavel
            prior = np.log(self.prior[idx])

            # aplica a formula gaussiana nos atributos
            pdf_valores = self.pdf(idx, x)

            # soma os log das probabilidades de cada atributo
            #+ 1e-18 eh pra nn quebrar no log
            posterior = np.sum(np.log(pdf_valores + 1e-18))

            # chance final se calcula por prior + posterior
            chance_final = prior + posterior
            #append = adiciona na lista de posteriores
            posteriors.append(chance_final)

        # argmax = retorna o indice do maior valor da lista de posteriores pra saber quem venceu
        return self.classes[np.argmax(posteriors)]

    # pdf = funcao de densidade de probabilidade, que serve para calcular a probabilidade de uma amostra pertencer a uma classe
    def pdf(self, classe_idx, x):
        media = self.media[classe_idx]
        variancia = self.variancia[classe_idx]

        #formula gaussiana = e^(-((x - media)**2) / (2 * variancia))/sqrt(2 * pi * variancia)

        #exp = calcula o numero de euler de qualquer valor dentro do parentese
        numerador = np.exp(-((x - media)**2) / (2 * variancia))
        denominador = np.sqrt(2 * np.pi * variancia)

        return numerador / denominador

def calcular_metricas(y_test, y_pred):
    vp = np.sum((y_pred == 1) & (y_test == 1)) #verdadeiro positivo = quantos o modelo disse ser 1 e eram 1 de verdade
    vn = np.sum((y_pred == 0) & (y_test == 0)) #verdadeiro negativo = quantos o modelo disse ser 0 e eram 0 de verdade
    fp = np.sum((y_pred == 1) & (y_test == 0)) #falso positivo = quantos o modelo disse ser 1 e na vdd eram 0
    fn = np.sum((y_pred == 0) & (y_test == 1)) #falso negativo = quantos o modelo disse ser 0 e na vdd eram 1

    acuracia = (vp + vn) / (vp + vn + fp + fn) #acuracia = porcentagem total de acertos
    precisao = vp / (vp + fp) if (vp + fp) > 0 else 0 #precisao = dos que o modelo chutou 1 (vp+fp), quantos eram de verdade
    recall = vp / (vp + fn) if (vp + fn) > 0 else 0 #recall = dpos que eram 1 msm, quantos o modelo encontrou
    f1_score = 2 * (precisao * recall) / (precisao + recall) if (precisao + recall) > 0 else 0 #f1-score = equilibrado entre precisao e recall (bom para dados desbalanceados)

    return acuracia, precisao, recall, f1_score

#validação cruzada
#majoritariamente usado para ver se o modelo não decorou as respostas (overfitting)
def validacao_cruzada_nb(X, y, k_folds=5):

    #embaralha as amostras com o permutation e dps fatia elas com o array_split
    total_amostras = len(X)
    indices_embaralhados = np.random.permutation(total_amostras)
    fatias = np.array_split(indices_embaralhados, k_folds)


    notas_acuracia = []
    notas_f1 = []
    notas_recall = []
    tempos_treino = []
    tempos_teste = []
    #ele roda k vezes (5) e a cada vez ele escolhe um montinho para ser amostra de teste(1/5) e o restante para treino(4/5 = k-1)
    for i in range(k_folds):
        print("Rodada", i + 1, "de", k_folds)

        indices_teste = fatias[i] #montinho que sobrou para teste
        indices_treino = []  #montinho escolhidopra treino
        for j in range(k_folds):
            if j != i:
                #extend = adiciona o conteudo de um array a outro array
                indices_treino.extend(fatias[j])

        X_treino = X[indices_treino]
        y_treino = y[indices_treino]
        # X = variavel independente, y = variavel dependente
        X_teste = X[indices_teste]
        y_teste = y[indices_teste]

        #a cada rodada ele cria um novo modelo
        modelo = NaiveBayesUnivariado()

        inicio_treino = time.time() #liga
        modelo.fit(X_treino, y_treino) #chama o fit para ler as k-1 fatias que sobraram pra calcular media e variancia
        fim_treino = time.time()#desliga

        inicio_teste = time.time()
        previsoes = modelo.predict(X_teste) #a gente passa a fatia que sobrou e que ele nn conhece aqui pa ver se ele presta msm, aqui ele joga na pdf(probabilidade de cada classe)
        fim_teste = time.time()

        #append = adiciona na lista
        tempos_treino.append(fim_treino - inicio_treino)
        tempos_teste.append(fim_teste - inicio_teste)

        acuracia, precisao, recall, f1 = calcular_metricas(y_teste, previsoes) #compara as previsoes com as respostas reais
        notas_acuracia.append(acuracia)
        notas_f1.append(f1)
        notas_recall.append(recall)
        print("Acuracia da rodada:", acuracia)

    #return tira a media de todas as notas para ser justo
    return {
        "acuracia_media": np.mean(notas_acuracia),
        "f1_medio": np.mean(notas_f1),
        "recall_medio": np.mean(notas_recall),
        "recall_desvio": np.std(notas_recall),
        "tempo_treino_medio": np.mean(tempos_treino),
        "tempo_teste_medio": np.mean(tempos_teste)
    }

print("Naive Bayes Univariado - Classificação (INICIO)")

dados_classificacao = np.genfromtxt("gina_prior.csv", delimiter=",")
print("dados usados para classificação",dados_classificacao)

#pega todas as colunas exceto a ultima (x = todos os atributos, y = classe)
X_classificacao = dados_classificacao[:, :-1]
#pega apenas a ultima coluna (y = classe)
y_classificacao = dados_classificacao[:, -1]
#usado para normalizar os dizer que os bicho de classe -1 são classe 0
y_classificacao[y_classificacao == -1] = 0

print("Linhas:", X_classificacao.shape[0]) #shape[0] = retorna os elementos da matriz, nesse caso colunas
print("Colunas:", X_classificacao.shape[1]) #shape[1] = retorna os elementos da matriz, nesse caso linhas

# sem hiperparametros por aaqui

# chama a validação cruzada
resultados_nb = validacao_cruzada_nb(X_classificacao, y_classificacao, k_folds=5)

print("resultados naive bayes univariado")
#acuracoa do KNN foi 95% aqui foi 58%, isso acontce por ele ser tão naive e por estarmos trabalhando com pixels que dependem um do outro, a acuracia vai pra merda
print("Acuracia media: ", resultados_nb["acuracia_media"])
##f1 é o equilibrio entre precisão e recall, esse valor mais alto que a acuracia eh pq mto provavelmente ele ta chutando mto positivo e errando os negativso
print("F1 media: ", resultados_nb["f1_medio"])
print("Recall medio: ", resultados_nb["recall_medio"])
#o tempo de treino eh mto rapido tbm pq ele não é um algoritmo complexo e são calculos simples (mediaa, variancia, prior, tudo linear)
print("Tempo de treino medio: ", resultados_nb["tempo_treino_medio"])
#o tempo de teste foi quase instantaneo por ele ser eager learner(ansioso), ele constroi o modelo matematico no treino então fica mto rapido
print("Tempo de teste medio: ", resultados_nb["tempo_teste_medio"])

#o naive bayes teve dificuldade na classificação pois ele eh tão naive e ignora os dados dependentes uns dos outros, no caso estamos trabalhando com pixels oq serio crucial dele fazer.
print("Naive Bayes Univariado - Classificação (FIM)")