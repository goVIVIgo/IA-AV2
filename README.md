# IA-AV2

Objetivos: comparar o desempenho preditivo e computacional de algoritmos em tarefas de classificação e regressão.

# Algoritmo K-Nearest Neighbors (KNN)
- baseado em instancias, isso faz dele lazy learning, não constroi modelos matematicos ou ajusta pesos, ele decora os dados de treino, por isso o fit dele eh super rapido, e nas fases de teste (predict) ele preve o resultado de uma amostra calculando a distancia dela para todos os outros pontos memorizados, depois descobrindo os k(vizinhos) mais proximos.
- **Distância Euclidiana:** faz um calculo em linha reta entre dois pontos. **distancia(x, y) = raiz(somatorio((xi - yi)^2))**
- **Distância de Manhattan:** soma a diferença absoluta(modulo) das coordenadas, funciona tipo em formato de grade/plano.  **distancia(x, y) = somatorio(|x_i - y_i|) modulo**
- **KNN - CLASSIFICAÇÃO:** previsão feita por **voto majoritario** (classe que mais aparece entre os vizinhos)
- **KNN - REGRESSÃO:** tira a **media** dos valores alvo dos vizinhos

# Datasets (bancos de dados utilizados)
- **Classificação - Gina Prior:** dataset para reconhecimento de digitos manuscritos (separados em impares e pares). 3468 instâncias e 784 atributos (representa os pixels da imagens)
- **Regressão - CPMP-2015:** dataset com foco em o tempo de execução de algoritmos de otimização de containers (Container Pre-Marshalling Problem). 2108 instâncias e 24 atributos.

# Métricas de Avaliação
**Classificação:** usam matriz de confusão para calcular os erros e os acertos das categorias
- verdadeiros positivos (VP - o modelo previu 1 e era 1)
- verdadeiros negativos (VN - previu 0 e era 0)
- falsos positivos (FP - previu 1 mas era 0)
- falsos negativos (FN - previu 0 mas era 1)
- usamos esses erros e acertos para calcular a **acuracia** (porcentagem total de acertos) e o **f1_score** (media que equilibra os **falsos positivos(Precisão)** e os **falsos negativos(Recall)**, em dados desbalanceados)

**Regressão:** não usa matriz de confusão, o calculo é feito pensando nos erros
- valor real (y): numero de vdd do dataset
- valor previsto (ŷ): chute/calculo do modelo
- erro/residuo: real - previsto
- eles são usados para calcular o **MAE (ERRO MEDIO ABSOLUTO)** mede as distancias absolutas entre os valores previstos e os reais, e o **r2(coefieciente de determinação)** indica o quanto o modelo conseguiu entender a logica dos dados, vai de **0 a 1, 1 eh otimo 0 eh uma merda**

# KNN - Validação Cruzada (K-Fold)
Majoritariamente usado para ver se o modelo não decorou as respostas (overfitting)
- **Classificação (Euclidiana):** acurácia = ~95.4% | f1_score = ~95.2% |treino medio = ~0.000002s | teste medio = ~6.5s
- **Classificação (Manhattan):** acurácia = ~94.3% | f1_score = ~93.9% |treino medio = ~0.000001s | teste medio = ~6.0s
- **Regressão (Euclidiana):** MAE =  1074.7 | R² =  0.149 | treino medio = 0.0000005s | teste medio = ~1.6s
- **Regressão (Manhattan):** MAE = 1066.8 | R² =  0.154 | treino medio = 0.0000005s | teste medio = ~1.5s

# Conclusões
- **CLASSIFICAÇÃO**: teve resultados mto **bons**, 95% em praticamente tudo, foi bem bom
- **Quando o KNN - CLASSIFICAÇÃO se sairia RUIM:** em um dataset com muitas colunas, ele iria quebrar ao calcular as distancias, e em um caso de dados SUPER desbalanceados por causa do voto da maioria ele iria so prever tudo errado tbm
- **REGRESSÃO**: foi bem **ruim** (R² foi 15% de 100%  e MAE foi mto alto (quase 18min))
- **Quando o KNN - REGRESÃO se sairia BEM:** com padrões lineares. Se tivese tentando comprar uma casa, o knn de regressão iria encontrar os vizinhos mais proximos e tirar a media de preço deles.
- Isso "prova" que o KNN tem sensibilidade a escalas de dados, os atributos maiores monopolizam o modelo, prejudicando o KNN em algo que nn fosse linear
- Sobre o **lazy learning**, os treinos (fit) foram praticamente nulos, por so armazenar na memoria, ai o custo computacinal foi todo para os testes (predict), praticamente confirmando que algoritmos que se baseiam em instancias nn são bons para dar respostas em tempo real com massas de dados mto densas

# Algoritmo Naive Bayes - Univariado
- ele eh um **earger learner (ansioso)**, oq significa que ele ao contrario do KNN, nn so guarda (na memoria) os dados no metodo **fit(treino)**, ele faz o calculo da **prior, da media e da variancia**, construindo o modelo matematico dele no fit msm, e depois descartando as instancias
- no metodo **predict(teste)**, ele usa a PDF(função de densidade probabilidade), para prever a chance do atributo se encaixar na curva da classe, e usa **log** para **evitar erro de underflow**, E a **previsão** eh feita escolhendo a **classe com a maior pontuação final**
- muito **naive (estupido)**, ele assume que os **atributos são independentes** entre si na hora de calcular as chances do modelo
- **Univariado:** assume que os dados seguem uma **curva do sino** (a distribuição normal)

# Datasets (bancos de dados utilizados)
- **Classificação - Gina Prior:** dataset para reconhecimento de digitos manuscritos (separados em impares e pares). 3468 instâncias e 784 atributos (pixels da imagem).

# Métricas de Avaliação
**Classificação:** usa matriz de confusão para calcular os erros e acertos das categorias
- verdadeiros positivos (VP): o modelo previu 1 e era 1
- verdadeiros negativo (VN): o modelo previu 0 e era 0
- falso positivo (FP): o modelo previu 1 e era 0
- falso negativo (FN): o modelo previu 0 e era 1
  
Oq foi calculado:

- **Acuracia:** porcentagem total de acertos
- **F1_Score:** equilibrio entre **precisãoe e recall**. Bom para ver se o modelo não esta enviesado e bom para dados desbalanceados

# Naive Bayes - Validação Cruzada (K-Fold)
Uso para ver se o modelo não esta decorando as respostas e cuspindo (overfitting), ele faz isso separando o dataset todo em FOLDS e usando cada parte separada para treino(K-1) e teste(oq sobrou) pelo menos uma vez

- **Classificação (Naive Bayes - Univariado):** acurácia = ~58.9% | f1_score = ~70.4% | treino medio = ~0.02s | teste medio = ~0.027s

# Conclusões
- **Classificação**: não foi bom, a acuracia foi de 58%, e isso acontece por causa de ele ser tão naive, ele assumiu que cada pixel era independente, mas pra formar os digitos os pixels dependiam incondicionalmente uns dos outros, por isso caiu tanto (diferente do KNN que ia pelos mais proximos), e o f1_score foi de 70%, por ele ter sido maior que a acuracia da a entender que ele so saiu chutando os valores mesmo (aqui os valores positivos), então podemos supor que ele no final acabou ficando enviesado.
- **Quando o Naive Bayes - Univariado se sairia BEM:** com problemas com **atributos** que são realmente **independentes** (ex: diagnosticos de sintomas isolados, analises de spam)
- Sobre ele ser **eager learner (ansioso)**, seu treino foi super rapido, quase instantaneo (0.02s) ja que eh so calculo simples de media e variancia, e seu teste foi quase instantaneo tbm (0.027s), ja que o modelo aqui não precisa varrer todo o dataset dnv, ele so pega os valores que a formula retornou pra ele (diferente do knn que demorava pra porra de 6s)

# Algoritmo Perceptron Simples

- tenta traçar uma reta (hiperplano) para separar perfeitamente os dados da classe 0 e os dados da classe 1
- eh earger learner, ou seja, ele faz os calculos tudo no fit, ajustadno os pesos e as taxas tudo bonitinho e depois descarta os dados originais para ir para o predict dele que eh bemm rapidinho pq ele praticamente so compara com a reta traçada no fit
- **treino (fit):** faz os calculos para ajustar a reta
  - inicia os pesos e os bias zerados ou aleatórios
  - roda o dataset todo no numero de epocas definidas (100 epocas = roda 100 vezes)
  - calcula a equação da reta: multiplica os pixels do dataset pelos pesos (np.dot) e soma os bias
  - passa o resultado da equação da reta pela função degrau (função de ativação) se for maior que zero = classe 1, se for negativo = classse 0
  - compara com o valor real do modelo, se ele errar usa a taxa de aprendizado multiplicada pelo erro para atualizar os pesos e dar uns puxões na reta pra direção certa, e atualiza os pesos e os bias tbm
  - no final vamos ter os pesos e os bias finais, e a reta que vamos passar no predict
- **teste (predict):** usa a reta pronta ja ajustada
  - pega as imagens novas que ele nunca viu na vida, multiplica pelos pesos finais e multiplica pelos bias finais tbm
  - passa pela função de ativação (função degrau) so para ver se eh classe 0 ou classe 1
  - eh bem rapidinho por nn fazer quase nenhum cálculo, a reta ja esta la e ele usa ela.
- **hiperparametros:**
  - **epocas:** quantas vezes ele vai rodar o dataset do começo ao fim fazendo calculos la para ajustar a reta. Poucas epocas causam underfitting (nn da tempo de terminar de ajustara reta), e mtas epocas causam alto gasto computacional desnecessario, pq elas ficam gastando "voltas" no dataset quando tudo ja ta perfeitamente ajustado
  - **taxa de aprendizado:** são os puxões que vão ser dados na reta para deixar ela ajustadinha bonitinha, eh calculado multiplicando a taxa pelo erro. Se a taxa for mto baixa, vão ser passos curtos e demorados, porém eh mais seguro de testar assim, se for mto grande pode ser instável, deixando a reta torta, acaba nn sendo mto ideal.

# Métricas de Avaliação
**Classificação:** usa matriz de confusão para calcular os erros e acertos das categorias
- verdadeiros positivos (VP): o modelo previu 1 e era 1
- verdadeiros negativo (VN): o modelo previu 0 e era 0
- falso positivo (FP): o modelo previu 1 e era 0 (deu alarme falso)
- falso negativo (FN): o modelo previu 0 e era 1 (deixou passar batido)

Oq foi calculado:

- **acuracia:** porcentagem total de acertos gerais do modelo
- **precisão:** dos que o modelo "chutou" que era classe 1 (VP + FP), quantos eram classe 1 de verdade (VP)
- **recall (sensibilidade):** dos que eram classe 1 de verdade (VP + FN), quantos o modelo conseguiu encontrar e acertar (VP)
- **especificidade:** dos que eram classe 0 de verdade no gabarito (VN + FP), quantos o modelo encontrou e acertou (VN)
- **F1_Score:** equilíbrio entre precisão e recall. Bom para ver se o modelo não está enviesado (se ele tiver recall alto mas precisão lixo, o F1 cai)

# Perceptron - Validação Cruzada (K-Fold)
Foi usada a melhor configuração achada nos testes (taxa de aprendizado = 0.01 e epocas = 100):
- **Classificação (Perceptron Simples):** acurácia = ~82.7% | f1_score = ~83.8% | precisão = ~83.2% | recall = ~81.7% | especificidade = ~82.2% | treino médio = ~1.08s | teste médio = ~0.0002s

# Conclusões
- **Classsificação:** teve um resultado bom/intermediário/mais pra la do que pra ca.
  - o Perceptron foi mto mto melhor do que o naive bayes justamente por ele não ser "naive", ele nn assumiu independencia condiciona. Ele analisou o peso de todos os pixels juntinhos na msm equação pra traçar a reta
  - porém ele nn conseguiu ser melhor que o KNN, mto provavelmente pq os desenhos dos pixels do dayaset gina prior não são perfeitamente separáveis por uma linha reta, o knn avalia a distancia e os vizinhos, por ser um dataset de pixels mto provavelmente ele se saiu melhor por isso.
- **quando o perceptron simples de sairia BEM:** quando o **dataset fosse 100% linearmene separável**, passando uma reta que separe os dados perfeitamente em dois lados, se fosse identificação de letras em preto e branco nos pixels, identificação de spam em emails (sim e não) e em portas lógicas simples, AND, OR, NAND E NOR
- **comparando hiperparâmetros (testes 1 a 5):** comprovamos que taxa alta + poucas épocas (teste 4) eh mto ruim para o modelo (acurácia caiu pra 76%) e que colocar 1000 epocas (teste 5)e uma taxa de aprendizado instável (0.5), so faz o treino demorar 10x mais (10s) para travar em 80%, tem que ir equilibrando tudo bem direitinho (teste 1)
- **earger learner (ansioso) vs lazy learner (preguiçoso):** como ele treinou os pesos e os bias no fit, quando chegou no predict com a reta ajustada foi quase instantaneo, nesse quesito ele foir realmente melhor que o knn, que demorava mto no predict por apenas armazenar os dados em memória no fit dele.

# Comparação Algoritmo de Classificação: KNN vs. Naive Bayes Univariado vs. Perceptron
- **KNN (K-Nearest Neighbors):** lazy learner (preguiçoso)
  - **treino(fit): ~0.000002s.** Não faz contas, apenas armazena a matriz de dados na memória.
  - **teste(predict): ~6.5s.** Muito lento, pois precisa calcular a distância contra todas as 3468 instâncias
  - **acurácia (acertos): ~95%** (muito muito bom)
  - **viés (f1_score): ~95%** (foi equilibrado/bom)
  - **análise do comportamente do algoritmo:** conseguiu capturar mto bem o formato dos numeros pq ele avalia as proximidades geometricas dos pixels juntinhos

- **Naive Bayes Univariado:** eager learner (ansioso)
  - **treino(fit): ~0.02s.** Constroi o modelo matematico todinho aqui (media, variancia, prior)
  - **teste(predict): ~0.027s.** Eh super rapido pq so aplica os valores na equação PDF
  - **acurácia (acertos): ~58.9%**(HORRÍVEL)
  - **viés (f1_score): ~70%**, enviesado, chuta mto positivo(1) e por isso erra mto negativo(0)
  - **análise do comportamente do algoritmo:** falhou miseravelmente pq ele eh mto burro e so assumiu que cada pixel era independente e fds

- **Perceptron Simples**: earger learner (ansioso)
  - **treino(fit):** ~1.08s. Mais lento que o naive bayes, pois ele roda o dataset inteiro várias vezes (épocas) ajustando os pesos da reta matemática através dos erros
  - **teste(predict):** ~0.0002s. foi o mais rapido de todos, como a reta já tá pronta, ele só faz uma multiplicação simples e joga na função degrau (função de ativação)
  - **acurácia (acertos):** ~82.7% (bom/intermediário)
  - **viés (f1_score):** ~83.8% (equilibrado, não ficou chutando uma classe so de qualquer jeito, dirferente do naive bayes)
  - **análise do comportamento do algoritmo:** foi mto melhor que o Naive Bayes porque analisou todos os pixels juntos na mesma equação, mas não foi melhor que o KNN porque os desenhos dos números no dataset não são perfeitamente separáveis por uma única linha reta (problema não linear)                                                      

- **Conclusão;** com isso, podemos deduzir que nn existe algoritmo ruim aqui, existe algoritmo certo para a analise certa, o KNN foi mto bom na acurácia pq as imagens são dados espaciais e o formato importa, ai o calculo da distancia do KNN eh perfeito pra isso, mas ele não foi mto bom no tempo da previsão(teste), JUSTAMENTE por ter que fazer esse calculo da distancia para cada pixel longe pra caralho, ja o Naive Bayes, foi horrivel na acurácia pela suposição besta de que as instancias eram independentes, mas em compensação amassou no tempo geral de execução ele foi mto bom pela caracteristica ansiosa ele de montar as cosia logo e deixar o terreno todo montado so pra chamar funções. O Perceptron foi como um perfeito meio termo, não foi tão bom quanto o KNN pela não linearidade do problema, mas sua acurácia foi boa e seu tempo de teste extremamente melhor, e  do naive bayes a gente nem fala pq ele foi humilhado kkkkkk, enfim ele eh um otimo meio termo pq ele eh acurado e rapido.

### Tabela Comparativa dos Modelos

| Classificador | Acurácia | Precisão | F1-Score | Tempo Treino (s) | Tempo Teste (s) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **K-vizinhos (Dist. Euclidiana)** | 0.96 ± 0.01 | 0.98 ± 0.00 | 0.95 ± 0.01 | 0.00 ± 0.00 | 5.91 ± 0.09 |
| **K-vizinhos (Dist. Manhattan)** | 0.95 ± 0.01 | 0.98 ± 0.01 | 0.95 ± 0.01 | 0.00 ± 0.00 | 5.58 ± 0.02 |
| **Bayesiano (Univariado)** | 0.58 ± 0.05 | 0.54 ± 0.03 | 0.70 ± 0.03 | 0.02 ± 0.01 | 0.03 ± 0.00 |
| **Perceptron Simples** | 0.82 ± 0.01  | 0.83 ± 0.02  | 0.83 ± 0.03  | 1.08 ± 0.03 | 0.00 ± 0.00
