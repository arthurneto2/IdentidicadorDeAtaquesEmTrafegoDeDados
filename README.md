# Identificador de Ataques de Tráfego de Dados

Este projeto implementa um sistema de detecção de ataques em tráfego de dados utilizando técnicas de Aprendizado de Máquina (Machine Learning). O modelo utiliza uma abordagem de **Ensemble Learning** (Votação Suave), combinando a robustez do *Random Forest* com a eficiência do *Histogram Gradient Boosting*.

## 📋 Funcionalidades

- Carregamento e pré-processamento de datasets de tráfego de rede.
- Treinamento de um classificador híbrido (Random Forest + HistGradientBoosting).
- Previsão de classes de ataque em novos dados de teste.
- Geração automática de arquivo de submissão (`submission_ensemble.csv`).

## 🛠️ Tecnologias Utilizadas

- **Python 3**
- **Pandas**: Manipulação e análise de dados.
- **Scikit-learn**: Construção, treinamento e avaliação dos modelos de ML.

## 🚀 Como Executar

### Pré-requisitos

Certifique-se de ter o Python instalado em sua máquina. Em seguida, instale as bibliotecas necessárias executando o seguinte comando no terminal:

```bash
pip install pandas scikit-learn
```

### Executando o Detector

1. Certifique-se de que os arquivos de dados `train.csv` e `test.csv` estejam no mesmo diretório do script.
2. Execute o script Python:

```bash
python detector_de_ataques.py
```

3. O script processará os dados, treinará o modelo e gerará o arquivo `submission_ensemble.csv` com os resultados das previsões.

## 📂 Estrutura do Projeto

- `detector_de_ataques.py`: Script principal contendo a lógica de treinamento e inferência.
- `train.csv`: Dataset utilizado para treinar o modelo (contém as features e a coluna `target`).
- `test.csv`: Dataset utilizado para gerar as previsões (contém apenas as features).
- `submission_ensemble.csv`: Arquivo de saída gerado pelo script contendo os IDs e as classes preditas.
