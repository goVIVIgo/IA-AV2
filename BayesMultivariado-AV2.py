import numpy as np
import time
#treino do modelo por classe
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