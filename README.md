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
- **Classificação (Euclidiana):** acurácia = ~95.4% | f1_score = ~95.2% |treino medio = ~0.000002s | teste medio = ~6.5s
- **Classificação (Manhattan):** acurácia = ~94.3% | f1_score = ~93.9% |treino medio = ~0.000001s | teste medio = ~6.0s
- **Regressão (Euclidiana):** MAE =  1074.7 | R² =  0.149 | treino medio = 0.0000005s | teste medio = ~1.6s
- **Regressão (Manhattan):** MAE = 1066.8 | R² =  0.154 | treino medio = 0.0000005s | teste medio = ~1.5s

# Conclusões
- **CLASSIFICAÇÃO**: teve resultados mto bons, 95% em praticamente tudo, foi bem bom
- **REGRESSÃO**: foi bem ruim (R² foi 15% de 100%  e MAE foi mto alto (quase 18min))
- Isso "prova" que o KNN tem sensibilidade a escalas de dados, os atributos maiores monopolizam o modelo, prejudicando o KNN em algo que nn fosse linear
- Sobre o **lazy learning**, os treinos (fit) foram praticamente nulos, por so armazenar na memoria, ai o custo computacinal foi todo para os testes (predict), praticamente confirmando que algoritmos que se baseiam em instancias nn são bons para dar respostas em tempo real com massas de dados mto densas

