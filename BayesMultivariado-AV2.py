# trabalho de IA AV2 
# aluna: Maria Eduarda Ricoy (2410454) 
# responsável por: bayes multivariado  e slides 

import numpy as np
import time
# treino do modelo por classe
def treinar_bayes(X_treino, y_treino):
    classes = np.unique(y_treino)
    medias = []
    inversas = []
    log_dets = []
    priors = []

    for classe in classes:
        dados_classe = X_treino[y_treino == classe]

        # media e covariancia de cada classe
        media = np.mean(dados_classe, axis=0)
        cov = np.cov(dados_classe, rowvar=False)

        # so pra matriz nao dar problema na inversa
        cov = cov + np.eye(cov.shape[0]) * 0.001

        _, log_det = np.linalg.slogdet(cov)

        medias.append(media)
        inversas.append(np.linalg.inv(cov))
        log_dets.append(log_det)
        priors.append(len(dados_classe) / len(X_treino))

    return classes, np.array(medias), np.array(inversas), np.array(log_dets), np.array(priors)

# previsao dos dados de teste
def prever_bayes(X_teste, classes, medias, inversas, log_dets, priors):
    resultados = []
    d = X_teste.shape[1]

    for x in X_teste:
        chances = []

        for i in range(len(classes)):
            diferenca = x - medias[i]
            termo = diferenca.T @ inversas[i] @ diferenca
            # probabilidade de pertencer a cada classe
            log_prob = (
                np.log(priors[i])
                - 0.5 * (d * np.log(2 * np.pi) + log_dets[i] + termo)
                )
            chances.append(log_prob)

        resultados.append(classes[np.argmax(chances)])

    return np.array(resultados)

# metricas da classificacao
def metricas(y_real, y_pred):
    vp = np.sum((y_real == 1) & (y_pred == 1))
    vn = np.sum((y_real == 0) & (y_pred == 0))
    fp = np.sum((y_real == 0) & (y_pred == 1))
    fn = np.sum((y_real == 1) & (y_pred == 0))

    acuracia = (vp + vn) / len(y_real)
    precisao = vp / (vp + fp) if (vp + fp) > 0 else 0
    recall = vp / (vp + fn) if (vp + fn) > 0 else 0
    f1 = 2 * precisao * recall / (precisao + recall) if (precisao + recall) > 0 else 0

    return acuracia, precisao, recall, f1
# validacao cruzada em 5 partes
def validacao_cruzada(X, y, k=5):
    np.random.seed(42)

    indices = np.random.permutation(len(X))
    folds = np.array_split(indices, k)

    acuracias = []
    precisoes = []
    recalls = []
    f1s = []
    tempos_treino = []
    tempos_teste = []

    for rodada in range(k):
        print("rodada", rodada + 1, "de", k)

        teste = folds[rodada]
        treino = np.concatenate([folds[i] for i in range(k) if i != rodada])

        X_treino = X[treino]
        y_treino = y[treino]
        X_teste = X[teste]
        y_teste = y[teste]

        inicio = time.time()
        classes, medias, inversas, log_dets, priors = treinar_bayes(X_treino, y_treino)
        tempos_treino.append(time.time() - inicio)

        inicio = time.time()
        y_pred = prever_bayes(X_teste, classes, medias, inversas, log_dets, priors)
        tempos_teste.append(time.time() - inicio)

        acuracia, precisao, recall, f1 = metricas(y_teste, y_pred)

        acuracias.append(acuracia)
        precisoes.append(precisao)
        recalls.append(recall)
        f1s.append(f1)

        print("acurácia:", acuracia)
        print("f1:", f1)

    return {
        "acuracia_media": np.mean(acuracias),
        "acuracia_desvio": np.std(acuracias),
        "precisao_media": np.mean(precisoes),
        "precisao_desvio": np.std(precisoes),
        "recall_medio": np.mean(recalls),
        "recall_desvio": np.std(recalls),
        "f1_medio": np.mean(f1s),
        "f1_desvio": np.std(f1s),
        "tempo_treino_medio": np.mean(tempos_treino),
        "tempo_treino_desvio": np.std(tempos_treino),
        "tempo_teste_medio": np.mean(tempos_teste),
        "tempo_teste_desvio": np.std(tempos_teste)
    }

#pega e carrega o dataset
dados = np.genfromtxt("gina_prior.csv", delimiter=",")

X = dados[:, :-1]
y = dados[:, -1]

y[y == -1] = 0

print("bayesiano multivariado - classificação")
print("linhas:", X.shape[0])
print("colunas:", X.shape[1])

# botar pra rodar
resultados = validacao_cruzada(X, y, k=5)

print("\nresultados")
print("acurácia:", resultados["acuracia_media"], "+/-", resultados["acuracia_desvio"])
print("precisão:", resultados["precisao_media"], "+/-", resultados["precisao_desvio"])
print("recall:", resultados["recall_medio"], "+/-", resultados["recall_desvio"])
print("f1-score:", resultados["f1_medio"], "+/-", resultados["f1_desvio"])
print("tempo de treino:", resultados["tempo_treino_medio"], "+/-", resultados["tempo_treino_desvio"])
print("tempo de teste:", resultados["tempo_teste_medio"], "+/-", resultados["tempo_teste_desvio"])

