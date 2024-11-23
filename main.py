import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


class ModeloExecutado():
    def __init__(self, X_train, X_test, Y_train, Y_test, Y_pred, model ):
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train 
        self.Y_test = Y_test
        self.Y_pred = Y_pred
        self.model = model
    

class Modelo():
    def __init__(self):
        pass

    def define_model(self, tipo_modelo:int):
        if tipo_modelo == 0:
            self.model = LinearRegression()
            self.tipo_modelo = tipo_modelo
        else:
            self.model = SVC()
            self.tipo_modelo = tipo_modelo

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
        self.df_encoded = pd.get_dummies(self.df, columns=['Species'], drop_first=False)
        
        # visualização do dataframe após One-Hot Encoding
        print("Dataframe após One-Hot Encoding de Species")
        print(self.df_encoded)
        print(f"Resumo dos dados")
        print(f"{self.df_encoded.describe(include='all')}")
        print("-"*80)
        
    def treinamento_regressao_linear(self):
 
        self.regr_lin_result = {}
        especies = ['Species_Iris-setosa', 'Species_Iris-versicolor', 'Species_Iris-virginica']
 
        for chave in especies:

            print(f"Executando regressão linear para espécie {chave}")

            # Variáveis independentes (X)
            # Avaliar flor que está na chave
            colunas_a_remover = [item for item in especies if item != chave]
            X = self.df_encoded.drop(columns=colunas_a_remover)
            
            # Variável dependente (Y) - precisamos de valores numéricos para a regressão
            Y = self.df_encoded[chave].astype(int)  # Convertendo para 0 ou 1
           
            # Dividir os dados em treino e teste
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

            self.model.fit(X_train, Y_train)

            # Prever os valores com o conjunto de teste
            Y_pred = self.model.predict(X_test)

            modelo_executado = ModeloExecutado(X_train, X_test, Y_train, Y_test, Y_pred, self.model)
            
            self.regr_lin_result[chave] = modelo_executado


    def teste_regressao_linear(self):
        
        # Avaliar o modelo
        
        for especie, modelo in self.regr_lin_result.items():
        
            mse = mean_squared_error(modelo.Y_test, modelo.Y_pred)
            print(f'Modelo: regressão linear, Espécie: {especie}, Mean Squared Error: {mse}')
        
    def treinamento_SVC(self):
        
        # Mapear os números de volta para as espécies (opcional)
        self.df['Species'] = self.df['Species'].map({'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2})   
        
        X = self.df.drop(columns=['Species'])
        Y = self.df['Species']
        
        # Dividir os dados em treino e teste
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        
        self.model.fit(X_train, Y_train)
        
        # Prever os valores com o conjunto de teste
        Y_pred = self.model.predict(X_test)

        self.svc = ModeloExecutado(X_train, X_test, Y_train, Y_test, Y_pred, self.model)

        
    def teste_SVC(self):
        mse = mean_squared_error(self.svc.Y_test, self.svc.Y_pred)
        print(f'Modelo: SVC, Mean Squared Error: {mse}')
        
        print('-'*80)
        
        Y_test = self.svc.Y_test.tolist()
        
        for i in range(0,len(Y_test)):
            print(f"i:{i} Igual: {Y_test[i]==self.svc.Y_pred[i]} Y_test:{Y_test[i]} Y_pred:{self.svc.Y_pred[i]}")
        
        
    def Treinamento(self):
        """
        Treina o modelo de machine learning.

        Detalhes:
            * Utilize a função `train_test_split` para dividir os dados em treinamento e teste.
            * Escolha o modelo de machine learning que queira usar. Lembrando que não precisa ser SMV e Regressão linear.
            * Experimente técnicas de validação cruzada (cross-validation) para melhorar a acurácia final.
        
        Nota: Esta função deve ser ajustada conforme o modelo escolhido.
        """
        
        if self.tipo_modelo == 0:
            self.treinamento_regressao_linear()
        else:
            self.treinamento_SVC()
        


    def Teste(self):
        """
        Avalia o desempenho do modelo treinado nos dados de teste.

        Esta função deve ser implementada para testar o modelo e calcular métricas de avaliação relevantes, 
        como acurácia, precisão, ou outras métricas apropriadas ao tipo de problema.
        """
        
        if self.tipo_modelo == 0:
            self.teste_regressao_linear()
        else:
            self.teste_SVC()
        

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

    #regressão linear   
    print("*"*80)
    print("MODELO REGRESSÃO LINEAR")
    print("*"*80)

    modelo1 = Modelo()   
    modelo1.define_model(0)
    modelo1.Train()
    modelo1.Teste()
    
    # SVC
    print("*"*80)
    print("MODELO SVC")
    print("*"*80)
    modelo2 = Modelo()   
    modelo2.define_model(1)
    modelo2.Train()
    modelo2.Teste()


if __name__ == "__main__":
    main()
    
    
