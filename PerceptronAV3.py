import numpy as np
import time

# Perceptron Simples - Eager Learner - Classificação
# tenta traças uma reta (hiperplano) que separa os dados da classe 0 dos dados da classe 1
# muito bom para problemas linearmente separáveisq,se os dados  forem mto misturados ele eh meio ruim

# eh earger learner = ele treina as coisas no fit, FAZ CALCULOS, puxando e empurrando a reta ate ela ficar boa
# (obs: depois de achar a reta os dados brutos sao descartados)
# fit:
# inicia os pesos e os bias zerados
# roda todo o dataset várias vezes (epocas, 100 epocas =  roda 100 vezes)
# para cada imagem, multiplica os pixels pelos pesos (np.dot) e soma com o bias
# passa pela função degrau (fnção de ativação), se for 0 ou positivo, chuta classe 1, se for negativo, chuta classe 0
# SE ele errar o chute dele (depois de comparar com o y real (y[idx])), ele usa a taxa de aprendizado multiplicada pelo tamanho do erro para ajustar os pesos e o bias

#predict
# usa a reta pronta do fit, ele apenas multiplica as imagens novas pelos pesos e soma com o bias final
# passa pela função degrau (fnção de ativação), se for 0 ou positivo, chuta classe 1, se for negativo, chuta classe 0
# eh mto mto mto rápido pq ele usa a reta pronta do fit, praticamente nn faz calculos.

#OBS: o objetivo do fit (-> compara gabrito e corrige reta) eh encontrar uma regra matemática geral (taxa de aprendizado, epocas) que seja bom o suficiente para que funcione
# em qualquer coisa que ele va ser aplicado no futuro (predict) (-> formatar a função degrau de uma imagem nova)

#obs.2: epocas = quantas vezes o algoritmo vai rodar no total o dataset do começo ao fim e fazer os calculos (o chute e o ajuste)
# epocas pequenas dms = underfitting pq ele nn tem tempo de se ajustar direito
# epocas grandes = vai ficar varrendo os dados sem melhorar nada, podendo ate ocorrer overfitting, e vai demorar mto tbm pra varrer as coisa do dataset
#obs.3: taxa de aprendizado = multiplica o erro pra decidir o quaão forte vai ser o puxão que ele vai dar na reta pra corrigir o erro
#taxas mto pequenas = o modelo roda todas as epocas e no final nn ajusta nada direito pq ele puxa muito pouco
#taxas mto grandes = o modelo roda todas as epocas e fica todo torto pq ele ajustou/saltou muito a reta e deixou ela toda bamba

class Perceptron:
    def __init__(self, taxa_aprendizado=0.01, epocas=100):
        # hiperparametros = taxa de aprendizado e epocas
        self.taxa_aprendizado = taxa_aprendizado
        self.epocas = epocas
        self.pesos = None
        self.bias = None 

    #o Perceptron eh earger learner, ele faz calculos no fit dele construindo o modelo matemático
    #objetivo principal: encontrar uma reta (hiperplano) que separe perfeitamente os dados da classe 0 dos dados da classe 1
    #dps de achar essa reta, os dados brutos são descartados
    # fit ajusta os pesos, em suma
    def fit(self, X, y):
        n_amostras, n_atributos = X.shape

        # inicias com os pesos e os bias zerados (podia ser aleatorio tbm)
        # cria um vetor de pesos zerados e um unico bias (viés)
        self.pesos = np.zeros(n_atributos)
        self.bias = 0

        # epocas definem quantas vezes o algoritmo vai rodar no total o dataset do começo ao fim
        # a cada iteração das epocas ele tenta ajusrar a reta (hiperplano) para corrigir os erros da varrdura anterior
        for _ in range(self.epocas):
            for idx, x_i in enumerate(X):
                # Equação da reta: (Atributos * Pesos) + Bias
                #pra cada imagem, ele aplica o np.dot (produto escalar entre vetores)
                #multiplica o valor de cada pixel do dataset pela força do seu peso correspondente e soma todos os resultados, adicionando o bias no final
                saida_linear = np.dot(x_i, self.pesos) + self.bias

                #passa o resultado da saida linear pela funcao de ativacao (função degrau, se der maior ou igual a 0, a classe é 1. Se der negativo, é classe 0.)
                y_previsto = self._funcao_degrau(saida_linear)

                #se o modelo acertou, não mexe em nada
                # seerrou, ele precisa aprender e se ajustar
                if y_previsto != y[idx]:
                    # calcula a diferema entre o valor esperado e o oq ele chutou
                    # ele pega essa diferenca e multiplica pela taxa de aprendizado (que define o tamanho do passo da correção)
                    atualizacao = self.taxa_aprendizado * (y[idx] - y_previsto)

                    # aplica a atualização (ajuste) aos pesos
                    self.pesos += atualizacao * x_i
                    #o bias também é atualizado
                    self.bias += atualizacao

                    #esse passo a passo todo vai ajustando os pesos e o bias para corrigir os erros ate que eles estejam corretos entre as duas classes
        return self


    def _funcao_degrau(self, x):
        # função de ativação: se o cálculo der maior ou igual a 0, a classe é 1. Se der negativo, é classe 0.
        return np.where(x >= 0, 1, 0)

    #o predict dele igual o do naive bayes eh bem rapidinho pq os calculos ja foram feitos no fit, ele sp faz a multiplicação de matrizes
    def predict(self, X):
        # na fase de teste, ele apenas aplica a equação da reta com os pesos já treinados pelo fit
        # para cada amostra nova do dataset que chega, ele multiplica pelos pesos finais qie foram treinados e soma com o bias final
        saida_linear = np.dot(X, self.pesos) + self.bias
        # passa o resultado da saida linear pela funcao de ativacao para definir a classe final (0 ou 1)
        y_previsto = self._funcao_degrau(saida_linear)
        return y_previsto

    #obs: não tem erro no predict pq oq ele aprendeu no fit meio que eh lei geral, ele não contraria isso em nenhum outro canto do algotimo
    #obs.2: toda vez que um numero aparecer a função degrau tem que aparecer tbm pra classificar ele corretamente (0 ou 1)


def calcular_metricas(y_test, y_pred):
    vp = np.sum((y_pred == 1) & (y_test == 1)) # verdadeiro positivo = quantos o modelo disse ser 1 e eram 1 de verdade
    vn = np.sum((y_pred == 0) & (y_test == 0)) # verdadeiro negativo = quantos o modelo disse ser 0 e eram 0 de verdade
    fp = np.sum((y_pred == 1) & (y_test == 0)) # falso positivo = quantos o modelo disse ser 1 e na vdd eram 0
    fn = np.sum((y_pred == 0) & (y_test == 1)) # falso negativo = quantos o modelo disse ser 0 e na vdd eram 1

    acuracia = (vp + vn) / (vp + vn + fp + fn) # acuracia = porcentagem total de acertos
    precisao = vp / (vp + fp) if (vp + fp) > 0 else 0 # precisao = dos que o modelo chutou 1, quantos eram de verdade
    recall = vp / (vp + fn) if (vp + fn) > 0 else 0 # recall (sensibilidade) = dos que eram 1 msm, quantos o modelo encontrou
    f1_score = 2 * (precisao * recall) / (precisao + recall) if (precisao + recall) > 0 else 0 # f1-score = equilibrado entre precisao e recall

    especificidade = vn / (vn + fp) if (vn + fp) > 0 else 0 # especificidade = dos que eram 0 msm, quantos o modelo encontrou

    matriz_confusao = ([[vn, fp],
                    [fn, vp]])

    return acuracia, precisao, recall,f1_score, especificidade, matriz_confusao

# Validação Cruzada
# Validação Cruzada
# majoritariamente usado para ver se o modelo não decorou as respostas (overfitting)
def validacao_cruzada_perceptron(X, y, taxa_aprendizado, epocas, k_folds=5):

    # embaralha as amostras com o permutation e dps fatia elas com o array_split
    total_amostras = len(X)
    indices_embaralhados = np.random.permutation(total_amostras)
    fatias = np.array_split(indices_embaralhados, k_folds)

    notas_acuracia = []
    notas_precisao = []
    notas_recall = []
    notas_especificidade = []
    notas_f1 = []
    tempos_treino = []
    tempos_teste = []
    matrizes_confusao = []

    # ele roda k vezes (5) e a cada vez ele escolhe um montinho para ser amostra de teste(1/5) e o restante para treino(4/5 = k-1)
    for i in range(k_folds):
        print(f"Rodada {i + 1} de {k_folds}...")

        indices_teste = fatias[i] # montinho que sobrou para teste
        indices_treino = [] # montinho escolhido pra treino
        for j in range(k_folds):
            if j != i:
                # extend = adiciona o conteudo de um array a outro array
                indices_treino.extend(fatias[j])

        X_treino = X[indices_treino]
        y_treino = y[indices_treino]
        # X = variavel independente, y = variavel dependente
        X_teste = X[indices_teste]
        y_teste = y[indices_teste]

        # a cada rodada ele cria um novo modelo do perceptron e passa as epocas e a taxa de aprendizado
        modelo = Perceptron(taxa_aprendizado=taxa_aprendizado, epocas=epocas)

        inicio_treino = time.time() # liga
        modelo.fit(X_treino, y_treino) # chama o fit para construir o hiperplano separador com as k-1 fatias
        fim_treino = time.time() # desliga

        inicio_teste = time.time() # liga
        previsoes = modelo.predict(X_teste) # a gente passa a fatia que sobrou e que ele nn conhece aqui pa ver se ele presta msm
        fim_teste = time.time() # desliga

        # append = adiciona na lista
        tempos_treino.append(fim_treino - inicio_treino)
        tempos_teste.append(fim_teste - inicio_teste)

        # compara as previsoes com as respostas reais para puxar acuracia, matriz e as outras metricas
        acuracia, precisao, recall, especificidade, f1, matriz = calcular_metricas(y_teste, previsoes)

        notas_acuracia.append(acuracia)
        notas_precisao.append(precisao)
        notas_recall.append(recall)
        notas_especificidade.append(especificidade)
        notas_f1.append(f1)
        matrizes_confusao.append(matriz)

        print("Acurácia da rodada:", acuracia)
        print("Precisão da rodada:", precisao)
        print("Recall da rodada:", recall)
        print("Especificidade da rodada:", especificidade)
        print("F1 da rodada:", f1)
        print("Matriz de confusão da rodada:", matriz)

    # return tira a media de todas as notas para ser justo
    return {
        "acuracia_media": np.mean(notas_acuracia),
        "acuracia_desvio": np.std(notas_acuracia),
        "precisao_media": np.mean(notas_precisao),
        "precisao_desvio": np.std(notas_precisao),
        "recall_media": np.mean(notas_recall),
        "recall_desvio": np.std(notas_recall),
        "especificidade_media": np.mean(notas_especificidade),
        "especificidade_desvio": np.std(notas_especificidade),
        "f1_medio": np.mean(notas_f1),
        "f1_desvio": np.std(notas_f1),
        "tempo_treino_medio": np.mean(tempos_treino),
        "tempo_treino_desvio": np.std(tempos_treino),
        "tempo_teste_medio": np.mean(tempos_teste),
        "tempo_teste_desvio": np.std(tempos_teste),
        # soma todas as matrizes de confusao
        "matriz_confusao_total": np.sum(matrizes_confusao, axis=0) # somar no axis=0 junta todas elas em uma matriz final de shape (2, 2).
    }

print("Perceptron Simples - Classificação (INICIO)")

dados_classificacao = np.genfromtxt(r"C:\Users\vivif\OneDrive\Desktop\TrabalhosSillys\SEXTO SEMESTRE\IA\AULA 02\trabalho AV2\gina_prior.csv", delimiter=",")
print("dados usados para classificação\n", dados_classificacao)

#pega todas as colunas exceto a ultima (x = todos os atributos, y = classe)
X_classificacao = dados_classificacao[:, :-1]
#pega apenas a ultima coluna (y = classe)
y_classificacao = dados_classificacao[:, -1]
#usado para normalizar os dizer que os bicho de classe -1 são classe 0
y_classificacao[y_classificacao == -1] = 0

print("Linhas:", X_classificacao.shape[0]) #shape[0] = retorna os elementos da matriz, nesse caso colunas
print("Colunas:", X_classificacao.shape[1]) #shape[1] = retorna os elementos da matriz, nesse caso linhas

print("TESTE 1: Perceptron Simples - Classificação - Taxa=0.01, Epocas=100")
# chama a validação cruzada
resultados_p1 = validacao_cruzada_perceptron(X_classificacao, y_classificacao, taxa_aprendizado=0.01, epocas=100, k_folds=5)

print("resultados perceptron simples (taxa 0.01)")
# acurácia de ~82.7%, foi muito melhor que o Naive Bayes (~58%) porque o Perceptron avalia a relação entre todos os pixels simultaneamente para traçar uma reta, ao invés de isolar elas
# foi menor quee o KNN (~95%), provando que o dataset Gina Prior não é perfeitamente separável por uma linha reta (não é linearmente separável)
print("Acuracia media: ", resultados_p1["acuracia_media"])
# F1-Score alto (~83.8%) e próximo da acurácia mostra um modelo equilibrado, que não está apenas chutando uma única classe, como ocorrreu no naive bayes
print("F1 media: ", resultados_p1["f1_medio"])

# treino demorou ~1.08s, pq ele precisou de tempo para varrer o dataset 100 vezes ajustando os pesos bem bonitinhos
print("tempo de treino medio: ", resultados_p1["tempo_treino_medio"])
# teste foi praticasmente instantâneo (~0.0002s), isso pois ele eh um Eager Learner: o modelo demora no treino para criar a equação ee quando chega no teste so aplica a formula
print("tempo de teste medio: ", resultados_p1["tempo_teste_medio"])

print("precisão media: ", resultados_p1["precisao_media"])
print("acuracia desvio: ", resultados_p1["acuracia_desvio"])
print("precisao desvio: ", resultados_p1["precisao_desvio"])
print("f1 desvio: ", resultados_p1["f1_desvio"])

# recall (~81.7%) e especificidade (~82.2%) muito próximos da a entender que o hiperplano (reta) ficou bem centralizado, acertando bem tanto as classes 1 quanto as classes 0
print("recall media: ", resultados_p1["recall_media"])
print("recall desvio: ", resultados_p1["recall_desvio"])
print("especificidade media: ", resultados_p1["especificidade_media"])
print("especificidade desvio: ", resultados_p1["especificidade_desvio"])

print("tempo de treino desvio: ", resultados_p1["tempo_treino_desvio"])
print("tempo de teste desvio: ", resultados_p1["tempo_teste_desvio"])

# matriz de confusão  total = equilíbrio: erros bem distribuídos entre falsos positivos (285) e falsos negativos (312), isso significa que a reta separou bem as classes
print("matriz de confusao total:", resultados_p1["matriz_confusao_total"])


print("TESTE 2: Perceptron Simples - Classificação - Taxa=0.1, Epocas=100")
# chama a validação cruzada
resultados_p2 = validacao_cruzada_perceptron(X_classificacao, y_classificacao, taxa_aprendizado=0.1, epocas=100, k_folds=5)

print("resultados perceptron simples (taxa 0.1)")
# com a taxa 10x maior, a acurácia caiu um pouco (~82%)
# isso acontece porque o algoritmo dá "passos muito largos" para corrigir o erro e acaba pulando o ajuste da reta
print("Acuracia media: ", resultados_p2["acuracia_media"])
print("F1 media: ", resultados_p2["f1_medio"])
# os tempos de treino (~1.03s) e teste (~0.0003s) continuam iguais ao Teste 1, pois a quantidade de cálculos e épocas (100) foi exatamente a mesma
print("Tempo de treino medio: ", resultados_p2["tempo_treino_medio"])
print("Tempo de teste medio: ", resultados_p2["tempo_teste_medio"])

print("precisão media: ", resultados_p2["precisao_media"])
print("acuracia desvio: ", resultados_p2["acuracia_desvio"])
print("precisao desvio: ", resultados_p2["precisao_desvio"])
print("f1 desvio: ", resultados_p2["f1_desvio"])

# o passo foi muito largo, a estabilidade foi prejudicada: o Recall caiu para ~75.4%, ouy seja, a reta ficou mal posicionada e começou a perder os verdadeiros positivos
print("recall media: ", resultados_p2["recall_media"])
print("recall desvio: ", resultados_p2["recall_desvio"])
print("especificidade media: ", resultados_p2["especificidade_media"])
print("especificidade desvio: ", resultados_p2["especificidade_desvio"])

print("tempo de treino desvio: ", resultados_p2["tempo_treino_desvio"])
print("tempo de teste desvio: ", resultados_p2["tempo_teste_desvio"])

# matriz mostra que o passo muito largo fez os falsos negativos dispararam para 413.
# o modelo errou muito mais ao dizer que imagens da classe 1 eram 0.
print("matriz de confusao total:\n", resultados_p2["matriz_confusao_total"])


print("TESTE 3: Perceptron Simples - Classificação - Taxa=0.01, Epocas=10")
# chama a validação cruzada
resultados_p3 = validacao_cruzada_perceptron(X_classificacao, y_classificacao, taxa_aprendizado=0.01, epocas=10, k_folds=5)

print("resultados perceptron simples (taxa 0.01, epocas 10)")
# a acuracia caiu para ~79.8%.
# isso foi praticamente underfitting: com apenas 10 épocas, o algoritmo não teve varreduras suficientes para terminar de ajustar a inclinação da reta, em suma faltou tempo
print("Acuracia media: ", resultados_p3["acuracia_media"])
# f1_score foi para ~68.3%, a reta parou de ser ajustada no meio do caminho e o modelo ficou desequilibrado
print("F1 media: ", resultados_p3["f1_medio"])

# o tempo de treino caiu para ~0.10, como foram rodadas 10 épocas em vez de 100, o modelo treinou 10 vezes mais rápido, mas a qualidade foi sacrificada
print("tempo de treino medio: ", resultados_p3["tempo_treino_medio"])
print("tempo de teste medio: ", resultados_p3["tempo_teste_medio"])

print("precisão media: ", resultados_p3["precisao_media"])
print("acuracia desvio: ", resultados_p3["acuracia_desvio"])
print("precisao desvio: ", resultados_p3["precisao_desvio"])
print("f1 desvio: ", resultados_p3["f1_desvio"])

print("recall media: ", resultados_p3["recall_media"])
print("recall desvio: ", resultados_p3["recall_desvio"])
print("especificidade media: ", resultados_p3["especificidade_media"])
print("especificidade desvio: ", resultados_p3["especificidade_desvio"])

print("tempo de treino desvio: ", resultados_p3["tempo_treino_desvio"])
print("tempo de teste desvio: ", resultados_p3["tempo_teste_desvio"])

# matriz do underfitting: 552 falsos positivos, o modelo provavelmente classificou um monte de zeros como se fossem um
print("matriz de confusao total:", resultados_p3["matriz_confusao_total"])

print("TESTE 4: Perceptron Simples - Classificação - Taxa=0.1, Epocas=10")
# chama a validação cruzada
resultados_p4 = validacao_cruzada_perceptron(X_classificacao, y_classificacao, taxa_aprendizado=0.1, epocas=10, k_folds=5)

print("resultados perceptron simples (taxa 0.1, epocas 10)")
# acuracia foi para ~76.6%, a pior coisa que poderia acontecer: passos largos demais (0.1) e tempo de menos (10 épocas)
print("Acuracia media: ", resultados_p4["acuracia_media"])
print("Acuracia media: ", resultados_p4["acuracia_media"])
print("F1 media: ", resultados_p4["f1_medio"])
print("tempo de treino medio: ", resultados_p4["tempo_treino_medio"])
print("tempo de teste medio: ", resultados_p4["tempo_teste_medio"])
# precisão alta (~92.4%) eh pq ele ficou chutando tudo, o modelo ficou com a reta tão torta que quase não previu a classe 1, então, nas raras vezes que chutou 1, ele acertou
print("precisão media: ", resultados_p4["precisao_media"])
print("acuracia desvio: ", resultados_p4["acuracia_desvio"])
print("precisao desvio: ", resultados_p4["precisao_desvio"])
print("f1 desvio: ", resultados_p4["f1_desvio"])

# recall foi ruim (~57.5%), essa configuração deixou o modelo todo tortinho, perdendo quase metade das imagens da classe 1
print("recall media: ", resultados_p4["recall_media"])
print("recall desvio: ", resultados_p4["recall_desvio"])
print("especificidade media: ", resultados_p4["especificidade_media"])
print("especificidade desvio: ", resultados_p4["especificidade_desvio"])

print("tempo de treino desvio: ", resultados_p4["tempo_treino_desvio"])
print("tempo de teste desvio: ", resultados_p4["tempo_teste_desvio"])
#matriz teve 727 falsos negativos, basicamente ignorou metade das imagens da classe juntar passos largos (0.1) com pouco tempo (10) "quebra" o algoritmo
print("matriz de confusao total:", resultados_p4["matriz_confusao_total"])

print("TESTE 5: Perceptron Simples - Classificação - Taxa=0.5, Epocas=1000")
# chama a validação cruzada
resultados_p5 = validacao_cruzada_perceptron(X_classificacao, y_classificacao, taxa_aprendizado=0.5, epocas=1000, k_folds=5)

print("resultados perceptron simples (taxa 0.5, epocas 1000)")
# acuracia travou em ~80.8%, ou seja não adianta colocar infinitas épocas (1000) se a taxa for gigante (0.5), o modelo fica "quicando" de um lado pro outro e nunca se aquieta num ponto bom
print("Acuracia media: ", resultados_p5["acuracia_media"])
print("F1 media: ", resultados_p5["f1_medio"])
# treino demorou ~10.4 segundos,as épocas foram multiplicadas por 10 (de 100 para 1000), o tempo de treino subiu na mesma proporção
print("tempo de treino medio: ", resultados_p5["tempo_treino_medio"])
print("tempo de teste medio: ", resultados_p5["tempo_teste_medio"])

print("precisão media: ", resultados_p5["precisao_media"])
print("acuracia desvio: ", resultados_p5["acuracia_desvio"])
print("precisao desvio: ", resultados_p5["precisao_desvio"])
print("f1 desvio: ", resultados_p5["f1_desvio"])
# Recall médio (~77.6%) ficou pior que o do Teste 1, então a reta ficou instável e nunca se centralizou exatamente, mesmo treinando 10 vezes mais
print("recall media: ", resultados_p5["recall_media"])
print("recall desvio: ", resultados_p5["recall_desvio"])
print("especificidade media: ", resultados_p5["especificidade_media"])
print("especificidade desvio: ", resultados_p5["especificidade_desvio"])

print("tempo de treino desvio: ", resultados_p5["tempo_treino_desvio"])
print("tempo de teste desvio: ", resultados_p5["tempo_teste_desvio"])

# matriz teve 383 falsos negativos e 280 falsos positivos, a taxa de 0.5 realmente foi instavel.
print("matriz de confusao total:", resultados_p5["matriz_confusao_total"])

# Perceptron sofreu menos com a dependência dos pixels do que o Naive Bayes, pois ele analisa o "peso" de todos os pixels juntos na hora de criar a equação da reta, saltando de ~58% (NB) para ~83.8% de acurácia.
# MAS ele não conseguiu ser melhor que os ~95% do KNN, o que prova que os pixels do Gina Prior não são separáveis por uma única reta reta (o problema não é 100% linearmente separável)
# MAS seu o tempo de teste (earger learner - 0.0002) foi bem melhor que os 6 segundos do KNN (lazy learner).
print("Perceptron Simples - Classificação (FIM)")