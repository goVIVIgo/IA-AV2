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
