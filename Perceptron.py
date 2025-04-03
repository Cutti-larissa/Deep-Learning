#Faça você mesmo #2
from sklearn.preprocessing import MinMaxScaler

#Carga do dataset do repositório do sklearn
iris = datasets.load_iris()

# Colocando no Pandas para filtrar
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df["target"] = iris.target

scaler = MinMaxScaler()

# Normalizando características
iris_df[iris.feature_names] = scaler.fit_transform(iris_df[iris.feature_names])

#vamos manter apenas as duas primeiras classes: Iris Setosa e Iris Versicolour
vetores_classe_0 = iris_df[iris_df["target"] == 0]
vetores_classe_1 = iris_df[iris_df["target"] == 1]


#removendo colunas, para deixar o problema bidimensional
remover = ['petal length (cm)', 'petal width (cm)']
vetores_classe_0 = vetores_classe_0.drop(columns = remover)
vetores_classe_1 = vetores_classe_1.drop(columns = remover)

#ajustando classes para operar com o perceptron
vetores_classe_0["target"] = - 1
vetores_classe_1["target"] = + 1

#colocando em um vetor numpy para facilitar
vetores = np.concatenate((vetores_classe_0.to_numpy(), vetores_classe_1.to_numpy()))
np.random.seed(42)
np.random.shuffle(vetores) #misturando as linhas

#Valores x da fronteira, apenas para poder visualizar
x_hiperplano = np.array([-1,2])

#chute inicial. No mundo real, seria aleatório
w =  np.array([1,-1])
bias = 1 #perguntar sobre esse bias
eta = 0.1

#Implemente aqui o algoritmo de treinamento
incorretos = vetores
prod_escalar = 0
size = len(incorretos)
while(size > 0):
  contador = 0
  for x in incorretos:
    contador = contador +1
    x_vetor = x[:-1]
    prod_escalar = w.dot(x_vetor)
    if((x[-1] > 0 and prod_escalar > 0) or (x[-1] < 0 and prod_escalar < 0)):
      size = size - 1
    else:
      w = w + eta
  imprimir_plano(vetores, w, bias, incorretos)
