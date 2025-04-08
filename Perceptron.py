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
bias = 1
eta = 0.1

#Implemente aqui o algoritmo de treinamento
continuar = 1
while(continuar<10):
  incorretos = []
  for x in vetores:
    x_vetor = x[:-1]
    x_vetor = np.insert(x_vetor, 0, 1)
    prod_escalar = w.dot(x_vetor)
    if((x[-1] > 0 and prod_escalar > 0) or (x[-1] < 0 and prod_escalar < 0)):
      continue
    else:
      w = w - eta #classe errada e classe correta multiplicadas sempre geram -1
      incorretos.insert(-1, x)
    imprimir_plano(vetores, w, bias, incorretos) //linha laranja provavelmente errada
  continuar += 1
  if(incorretos is None):
    continuar = 10

#def ta_certo(dado, w, class_):
#  dado = dado[:2]
#  dado = np.insert(dado, 0, 1)
#  result = w.dot(dado)
#  if result >= 0 and class_ == 1:
#    return True
#  elif result < 0 and class_ == 0:
#    return True
#  return False

#def catar_incorretos(vetores, w, class_):
#  incorretos = []
#  for vetor in vetores:
#    if ta_certo(vetor, w, class_):
#      continue
#    else:
#      incorretos.append(vetor)
#    return incorretos

#classes = [int(x[2]) for x in vetores]

#epochs = 100
#for epoch in range(epochs):
#  for i, dado in enumerate(vetores):
#      variavel = ta_certo(dado, w, i)    
#      if variavel:
#        continue
#      else:
#        w = w - eta 
#  incorretos = catar_incorretos(vetores, w, classes)
#  print(incorretos)
#  imprimir_plano(vetores, w, bias, incorretos)
