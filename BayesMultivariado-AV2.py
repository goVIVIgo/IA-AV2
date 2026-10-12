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

        #so pra matriz nao dar problema na inversa
        cov = cov + np.eye(cov.shape[0]) * 0.001

        _, log_det = np.linalg.slogdet(cov)

        medias.append(media)
        inversas.append(np.linalg.inv(cov))
        log_dets.append(log_det)
        priors.append(len(dados_classe) / len(X_treino))

    return classes, np.array(medias), np.array(inversas), np.array(log_dets), np.array(priors)

# previsao dos dadoss de teste
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

# metricas das classificacao
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