pd.options.mode.chained_assignment = None

#Carga do dataset do repositório do sklearn
iris = datasets.load_iris()

# Colocando no Pandas para filtrar
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df["target"] = iris.target

#vamos manter apenas as duas primeiras classes: Iris Setosa e Iris Versicolour
vetores_classe_0 = iris_df[iris_df["target"] == 0]
vetores_classe_1 = iris_df[iris_df["target"] == 1]

#removendo colunas, para deixar o problema bidimensional
remover = ['petal length (cm)', 'petal width (cm)']
vetores_classe_0.drop(columns = remover,inplace = True)
vetores_classe_1.drop(columns = remover,inplace = True)

#ajustando classes para operar com o perceptron
vetores_classe_0["target"] = - 1
vetores_classe_1["target"] = + 1

#colocando em um vetor numpy para facilitar
vetores = np.concatenate((vetores_classe_0.to_numpy(), vetores_classe_1.to_numpy()))

#use KFold paraf fazer um teste de 5 folds. Mostre a matriz de confusão final, e a acurácia média
kf = KFold()
#iris espera - 1
#versicolor espera 1
irisC = 0
irisE = 0
versiC = 0
versiE = 0 
contagem = 0
for train_index, test_index in kf.split(vetores):
  classifier = Perceptron(eta0 = 0.05)
  treino = np.empty(len(train_index))
  for i in train_index:
    for j, x in enumerate(vetores):
      if(i == j):
        treino = np.insert(treino, -1, x) #rever
    treino = vetores[i] #a matriz de treino recebe o vetor de treino de indice i
  classifier.fit(treino[:,[0,1]], treino[:,2])
  teste = []
  for j in test_index:
    teste = vetores[i] #a matriz de teste recebe o vetor de teste de indice i
  predictions = classifier.predict(teste[:, [0, 1]])
  for i in teste:
    if (i[2] == predictions[contagem]):
      if(predictions[contagem] > 0):
        versiC += 1
      else:
        irisC += 1
    else:
      if (predictions[contagem] > 0):
        irisE += 1
      else:
        versiE += 1
    contagem += 1  
  #construir matriz de confusão 
