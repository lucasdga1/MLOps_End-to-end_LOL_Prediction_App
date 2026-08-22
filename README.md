# MLOps end-to-end LOL match prediction
* Acesse o app no seguinte URL:
    * <a href="site">LOL Match Prediction App</a>
# Objetivo
* O objetivo principal deste projeto é criar um app que possa ser utilizado por outros para prever resultados de partidas reais de LOL
a partir de dados estatísticos da partida.

# Dataset e Códigos
* O dataset foi fornecido pela EBAC como desafio acadêmico do último módulo do curso profissionalizante.
* Os dados para testar no app podem ser encontrados na pasta `src/match/app_test.csv`
    * Para fazer o teste com outras partidas você pode usar o arquivo `src/match/scrape.ipynb`. Alterando a url para a partida desejada.
* Os códigos dos notebooks utilizados está na pasta `notebooks`
* Os códigos das funções utilizadas está `src` sendo:
    * `src/api` o arquivo do FastAPI para o backend do app;
    * `src/feature_pipeline` o arquivo para limpeza e preprocessamento dos dados;
    * `src/inference_pipeline` o arquivo para realização das previsões;
    * `src/training_pipeline` os arquivos para treinamento e registro do modelo de aprendizado de máquina.

# Tecnologias Utilizadas
* **Python**: Linguagem de programação utilizada para análise de dados e machine learning.
* **Pandas**: Biblioteca para manipulação e análise de dados.
* **NumPy**: Biblioteca para operações numéricas.
* **Plotly**: Biblioteca para visualização de dados.
* **Scikit-learn e TensorFlow**: Biblioteca para machine learning, utilizada para aplicação dos algoritmos.
* **Docker**: Programa para a criação de contêineres e imagens dos códigos e programas utilizados;
* **Airflow**: Tecnologia que faz a orquestração e automatização de tarefas;
* **GoogleBigQuery**: Armazenamento dos dados a serem utilizados para atualizar e retreinar o modelo;
* **MLflow**: Biblioteca para registro dos modelos e seus parâmetros;
* **Optuna**: Biblioteca de ajuste de hiperparâmetros;
* **SHAP**: Biblioteca que mostra o quanto cada variável influencia modelo;
* **FastAPI**: Framework responsável por criar APIs;
* **Railway**: Plataforma para publicação de API, sites e apps;
* **Streamlit**: Framework para a criação de web apps interativos.
* **Github Actions**: Responsável pelo Continuous integration.

# Metodologia
1. Os dados foram carregados como dataframe no Pandas e avaliados quanto à sua integridade e pré-processamento;
2. Foram feitas análises com gráficos para tentar obter grau de correlação entre as variáveis;
3. Após alguns testes e análises foi decidido não tentar remover outliers dos dados, que podem fazer parte da dinâmica da partida.;
4. A base de dados foi separada em treino e teste;
5. A base de treino foi padronizada com Standard Scaler (para os modelos que precisam dessa etapa);
6. Foram treinados modelos de Regressão Logística, SVM, Random Forest, XGBoost e de redes neurais;
7. Os modelos foram testados e avaliados;
8. Foram criados arquivos em Python com funções para automatização de tarefas;
9. O modelo com melhor métrica foi retreinado, ajustado e registrado no MLflow;
10. Foram criados os contêineres. As tarefas foram estabelecidas e testadas no Airflow;
11. O arquivo de API foi criado no Docker para poder consumir o modelo mais atualizado;
12. O app foi criado usando streamlit;
13. Foi feito o CI via Github Actions;
14. A API foi exposta com Railway;
15. O app foi publicado no streamlit.io


# Análise de dados pré-modelagem
* Como os gráficos de análise exploratória demonstram que apesar de haver um valor, ou fator, em cada feature no qual 
a partida tende para uma equipe, o fato de não ser um volume significativo dificulta a previsão. Isso leva a crer que 
para alcançar a vitória em partidas competitivas é necessário mais do que estatísticas favoráveis.

# Treinamento
* A base de dados foi separada em treino e teste;
* Os dados de treino foram padronizados com Standard scaler;
* Em um primeiro momento o ajuste de hiperparâmetros foi realizado com Random Search, mas para o registro e utilização do
modelo no app foi utilizado a Optuna;

# Teste
* Os modelos foram testados e, como medidas de avaliação, foram obtidos os relatórios de classificação, matriz de confusão e curva AUC_ROC;
* A métrica alvo foi o Recall, devido à necessidade de identificar as partidas em que as equipes venceram
como verdadeiros positivos.

# Resultados
## Rede Neural
* O modelo de redes neurais obteve um recall de 0.74

## Random Forest
* O modelo de random forest obteve um recall de 0.71

## XGBoost
* O modelo de XGBoost obteve um recall de 0.71 também, mas obteve o maior número de acertos nas previsões em que a equipe
azul venceu, e, por isso, foi escolhida.

## Curva AUC_ROC
<img width="1280" height="720" alt="AUC_ROC" src="https://github.com/user-attachments/assets/aad2f656-14e4-4ea6-8f61-f7b0987385c5" />

## Análise SHAP de importância das features
* Parece que a variável mais determinante para o modelo na hora de prever os resultados das partidas foi a quantia de ouro
da equipe azul, seguido pela quantia de dragões e pela quantia de experiência dos heróis.

# Como o modelo pode auxiliar
* A partir da análise de importância das variáveis é possível perceber que não existe um fator único que determina o resultado,
mas é possível inferir se a performance deixou a desejar após considerar os dados.


# Contribuições
* Contribuições são bem-vindas! Sinta-se à vontade para abrir uma issue ou enviar um pull request.

# Autoria
Lucas Danziger Guimarães de Andrade

# Contato
<a href="mailto:lucas.dga1@gmail.com">Me envie um email<a/>
