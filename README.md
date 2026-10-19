<p align="center">
  <img src="images/Cabecalho.png" alt="Cabeçalho" width="100%">
</p>

<h1 align="center">
   Tiamat — <i>Classificação de Resposta Biológica Molecular</i>
</h1>

<p align="center">
  <i>Aprendizado de máquina para predição de respostas biológicas a partir de descritores moleculares, com otimização de hiperparâmetros por Optuna</i>
  <br><br>
  Ilum — School of Science | Campinas, 2026
</p>


## Sobre o projeto

O **Projeto Tiamat** é o trabalho final desenvolvido para a disciplina de Aprendizado de Máquina da Ilum — Escola de Ciência. O projeto tem como objetivo investigar, implementar e comparar diferentes algoritmos de aprendizado de máquina supervisionado para a **classificação binária de respostas biológicas de moléculas**, utilizando informações químicas representadas por descritores moleculares.

Para isso, utiliza-se o conjunto de dados **BioResponse**, disponibilizado na plataforma OpenML (ID: 4134). Cada amostra representa uma molécula descrita por um conjunto de variáveis numéricas, associadas a uma resposta biológica experimental categorizada em duas classes.

O trabalho busca compreender como diferentes arquiteturas de classificação se comportam diante desse problema, considerando tanto o desempenho preditivo quanto a capacidade de generalização para amostras não utilizadas durante o treinamento.

## Objetivos

- Investigar as características estatísticas e estruturais do conjunto de dados por meio de análise exploratória.
- Implementar um modelo *baseline* para estabelecer uma referência inicial de desempenho.
- Desenvolver e comparar diferentes algoritmos de classificação supervisionada.
- Otimizar os hiperparâmetros dos modelos por meio da biblioteca **Optuna**.
- Aplicar validação cruzada estratificada para avaliar a estabilidade do desempenho durante o treinamento.
- Comparar os modelos em um conjunto de teste independente, utilizando métricas complementares de avaliação.

## Metodologia

O desenvolvimento do projeto está organizado em etapas complementares:

1. **Análise exploratória de dados (EDA):** investigação da distribuição das variáveis, da resposta biológica e das relações entre os descritores moleculares.
2. **Redução de dimensionalidade:** aplicação da Análise de Componentes Principais (PCA) para explorar a estrutura dos dados em um espaço de menor dimensionalidade.
3. **Modelo de referência (*baseline*):** construção de uma referência inicial para contextualizar os resultados dos modelos mais complexos.
4. **Treinamento e otimização:** implementação dos classificadores selecionados e busca de hiperparâmetros com Optuna, utilizando até 100 *trials* por arquitetura.
5. **Validação cruzada:** utilização de `StratifiedKFold` para preservar, tanto quanto possível, a proporção das classes em cada partição.
6. **Avaliação final:** comparação dos modelos no conjunto de teste independente, com análise de métricas e curvas ROC.

A separação entre treinamento e teste é importante para reduzir o risco de estimativas excessivamente otimistas do desempenho preditivo. A otimização de hiperparâmetros deve ocorrer exclusivamente com os dados de treinamento, mantendo o conjunto de teste reservado para a avaliação final.

## Métricas de avaliação

Para caracterizar diferentes aspectos do desempenho dos classificadores, são consideradas métricas como:

- **Acurácia (*Accuracy*):** proporção total de classificações corretas.
- **Precisão (*Precision*):** proporção de previsões positivas que correspondem a casos positivos reais.
- **Revocação (*Recall*):** proporção de casos positivos reais identificados pelo modelo.
- **F1-score:** média harmônica entre precisão e revocação.
- **ROC-AUC:** medida da capacidade de discriminação entre as classes ao longo de diferentes limiares de decisão.

A análise conjunta dessas métricas permite uma comparação mais abrangente do que a utilização isolada da acurácia, especialmente quando as classes apresentam distribuições diferentes.

## Tecnologias utilizadas

| Biblioteca | Aplicação |
|---|---|
| `Python` | Linguagem de programação utilizada no projeto |
| `pandas` | Manipulação e organização dos dados |
| `NumPy` | Operações numéricas e computação vetorizada |
| `Matplotlib` | Construção de gráficos e visualizações |
| `Seaborn` | Visualização estatística dos dados |
| `scikit-learn` | Pré-processamento, treinamento, validação e avaliação dos modelos |
| `Optuna` | Otimização de hiperparâmetros |
| `XGBoost` | Implementação de modelos de *gradient boosting* |
| `LightGBM` | Implementação de modelos eficientes de *gradient boosting* |
| `SciPy` | Ferramentas para estatísticas |

## Requisitos e instalação

Para executar os notebooks, recomenda-se utilizar ``Python 3.10`` ou superior, com o Jupyter Notebook ou JupyterLab instalado.

Instale as dependências principais executando:

```bash
pip install pandas numpy matplotlib seaborn optuna scikit-learn xgboost lightgbm scipy jupyter
```

Para iniciar o ambiente JupyterLab, utilize:

```bash
jupyter lab
```

Em seguida, abra o notebook correspondente ao projeto e execute as células na ordem apresentada.

> **Nota:** a disponibilidade de recursos computacionais e o tempo de execução podem variar conforme o número de modelos, a configuração da otimização e o ambiente utilizado.

## Conjunto de dados

O projeto utiliza o conjunto **BioResponse**, disponível publicamente no OpenML.

- **Plataforma:** OpenML
- **Identificador:** 4134
- **Tipo de problema:** classificação binária supervisionada
- **Representação das entradas:** descritores moleculares numéricos
- **Variável-alvo:** resposta biológica experimental

A utilização desses dados permite investigar a aplicação de métodos de aprendizado de máquina à predição de respostas biológicas a partir de propriedades químicas moleculares.

**Acesso ao conjunto de dados:** [BioResponse — OpenML](https://www.openml.org/d/4134)

## Professor

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/drcassar">
        <img src="https://avatars.githubusercontent.com/u/9871905?v=4"
             width="100px;"
             alt="Perfil de Daniel R. Cassar no GitHub"
             style="border-radius:50%;"><br>
        <b>Prof. Dr. Daniel R. Cassar</b>
      </a>
    </td>
  </tr>
</table>

---

## Autor

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/FipaLipe">
        <img src="https://avatars.githubusercontent.com/u/178325127?v=4"
             width="100px;"
             alt="Perfil de Abraham Contreras no GitHub"
             style="border-radius:50%;"><br>
        <b>Abraham Contreras</b>
      </a>
    </td>
  </tr>
</table>

<p align="center">
  <img src="images/footer.png" alt="Rodapé" width="100%">
</p>
```
