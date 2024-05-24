# Execução do algoritmo

## Pré-processamento

* Classe: `PoiCategorizationDomain()`
* Método: `poi_gnn_adjacency_preprocessing`

Inicialmente é feito um pré-processamento nos arquivos (matrizes e afins), que faz o seguinte:

Itera sobre uma lista de ids (provavelmente IDs de usuários) e realiza as seguintes operações para cada usuário:

* Carrega diferentes matrizes e dados relacionados ao usuário atual.

* Realiza alguns cálculos e transformações nessas matrizes, como redimensionamento e normalização.
  * filtro de usuários com poucos check-ins

## Verifica Matrizes

* Classe: `PoiCategorizationJob()`
* Método: `matrices_verification`

Depois são feitas verificações nas matrizes para assegurar que as matrizes tem o mesmo tamanho

## Separação em conjuntos de teste e treinamento

* Classe `PoiCategorizationDomain()`
* Métodos: `k_fold_split_train_test`, `k_fold_with_replication_train_and_evaluate_model` e `train_and_evaluate_model`

É importante considerar e entender os arquivos gerados para executar o algoritmo, pois eles contém uma coluna
*category*, que serve como label para os dados e também como informação no aprendizado supervisionado

`k_fold_split_train_test`
- indica que será utilizada a abordagem de validação cruzada para o conjunto de dados
- realiza a validação cruzada para os dados de dias da semana, finais de semana e considerando toda a semana

`k_fold_with_replication_train_and_evaluate_model`
- executa o treinamento e a avaliação de modelos usando a técnica de k-fold com replicação
- itera sobre os folds e replicações, chama o método train_and_evaluate_model para treinar e avaliar cada modelo e armazena as informações relevantes
- ao final seleciona o melhor modelo com base na acurácia

`train_and_evaluate_model`
- conjunto de treinamento é o conjunto de dados de treinamento separado pela validação cruzada
- conjunto de teste é o conjunto de dados de teste separado pela validação cruzada
- dados utilizados para teste e treinamento: lista das matrizes dos arquivos gerados pelo pré-processamento
  - adjacency, temporal, adjacency_week, temporal_week, distance, duration, location_time, location_location
  - vale lembrar que cada linha na matriz corresponde a um usuário
  - label: lista de categorias visitadas pelo usuário
  - saída: prevê essa lista para o conjunto de teste
    - obs: a saída é uma lista contendo todas as categorias dos usuários, em sequência. Não foi feito
    um processamento para atribuir a saída na base de dados dos usuários, ou seja, não foi feita a rotulagem
    da base