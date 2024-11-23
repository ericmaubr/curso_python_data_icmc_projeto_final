import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression

class Modelo():
    def __init__(self):
        pass

    def CarregarDataset(self, path):
        """
        Carrega o conjunto de dados a partir de um arquivo CSV.

        Parâmetros:
        - path (str): Caminho para o arquivo CSV contendo o dataset.
        
        O dataset é carregado com as seguintes colunas: SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm e Species.
        """
        names = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']
        self.df = pd.read_csv(path, names=names)
        
        print(f"Quantidade de linhas carregadas: {len(self.df)}")
        print("-"*80)
        print(f"Resumo dos dados")
        print(f"{self.df.describe(include='all')}")
        print("-"*80)
        
        
        # apresenta um gráfico 3d com  SepalLengthCm, SepalWidthCm, PetalLengthCm
        cores = ['red' if s == 'Iris-setosa' else 'green' if s == 'Iris-versicolor' else 'blue' for s in self.df['Species']]
   
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_title("SepalLengthCm, SepalWidthCm, PetalLengthCm")
        ax.scatter(self.df['SepalLengthCm'], self.df['SepalWidthCm'], self.df['PetalLengthCm'], c = cores)
        plt.show()
        
        # apresenta um gráfico 3d com  SepalLengthCm, SepalWidthCm, PetalWidthCm
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_title("SepalLengthCm, SepalWidthCm, PetalWidthCm")
        ax.scatter(self.df['SepalLengthCm'], self.df['SepalWidthCm'], self.df['PetalWidthCm'], c = cores)
        plt.show()
        
        
    def TratamentoDeDados(self):
        """
        Realiza o pré-processamento dos dados carregados.

        Sugestões para o tratamento dos dados:
            * Utilize `self.df.head()` para visualizar as primeiras linhas e entender a estrutura.
            * Verifique a presença de valores ausentes e faça o tratamento adequado.
            * Considere remover colunas ou linhas que não são úteis para o treinamento do modelo.
        
        Dicas adicionais:
            * Explore gráficos e visualizações para obter insights sobre a distribuição dos dados.
            * Certifique-se de que os dados estão limpos e prontos para serem usados no treinamento do modelo.
        """
        
        # remove linhas em vazias
        # apesar de não haver linhas com dados vazios, é uma boa prática
        self.df = self.df.dropna()
        
        # aplica One-Hot Encoding na coluna Species
        # Removendo a primeira coluna. Ideal para regressão linear para evitar multicolinearidade.
        self.df_hot_encoding = pd.get_dummies(self.df, columns=['Species'], drop_first=True)
        
        # visualização do dataframe após One-Hot Encoding
        print("Dataframe após One-Hot Encoding de Species")
        print(self.df_hot_encoding)

    def Treinamento(self):
        """
        Treina o modelo de machine learning.

        Detalhes:
            * Utilize a função `train_test_split` para dividir os dados em treinamento e teste.
            * Escolha o modelo de machine learning que queira usar. Lembrando que não precisa ser SMV e Regressão linear.
            * Experimente técnicas de validação cruzada (cross-validation) para melhorar a acurácia final.
        
        Nota: Esta função deve ser ajustada conforme o modelo escolhido.
        """
        
        modelo = LinearRegression()
        
        X = np.array(self.df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']])
        Y = np.array(self.df['Species'])



    def Teste(self):
        """
        Avalia o desempenho do modelo treinado nos dados de teste.

        Esta função deve ser implementada para testar o modelo e calcular métricas de avaliação relevantes, 
        como acurácia, precisão, ou outras métricas apropriadas ao tipo de problema.
        """
        pass

    def Train(self):
        """
        Função principal para o fluxo de treinamento do modelo.

        Este método encapsula as etapas de carregamento de dados, pré-processamento e treinamento do modelo.
        Sua tarefa é garantir que os métodos `CarregarDataset`, `TratamentoDeDados` e `Treinamento` estejam implementados corretamente.
        
        Notas:
            * O dataset padrão é "iris.data", mas o caminho pode ser ajustado.
            * Caso esteja executando fora do Colab e enfrente problemas com o path, use a biblioteca `os` para gerenciar caminhos de arquivos.
        """
        self.CarregarDataset("iris.data")  # Carrega o dataset especificado.

        # Tratamento de dados opcional, pode ser comentado se não for necessário
        self.TratamentoDeDados()

        self.Treinamento()  # Executa o treinamento do modelo

# Lembre-se de instanciar as classes após definir suas funcionalidades
# Recomenda-se criar ao menos dois modelos (e.g., Regressão Linear e SVM) para comparar o desempenho.
# A biblioteca já importa LinearRegression e SVC, mas outras escolhas de modelo são permitidas.


def main():
    
    modelo1 = Modelo()
    
    modelo1.Train()
    
    



if __name__ == "__main__":
    main()
    
    
