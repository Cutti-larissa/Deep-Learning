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
#iris (-1) versi (1)
irisC = 0
irisE = 0
versiC = 0
versiE = 0

kf = KFold(n_splits=5, shuffle=True)
for train_index, test_index in kf.split(vetores):
  classifier = Perceptron(eta0 = 0.05)
  treino = [[0.0,0.0,0.0]] #ultimo vetor de treino
  treino = np.array(treino)
  teste = [[0.0,0.0,0.0]] #ultimo vetor de teste
  teste = np.array(teste)
  for i in train_index:
    for j, x in enumerate(vetores):
      if(i == j):
        treino = np.insert(treino, -1, x, axis = 0) #insere x no treino
  treino = np.delete(treino, -1, axis = 0)
  classifier.fit(treino[:,[0,1]], treino[:,2])
  for k in test_index:
    for l, y in enumerate(vetores):
      if(k == l):
        teste = np.insert(teste, -1, y, axis = 0) #insere y no teste
  teste = np.delete(teste, -1, axis = 0)
  predictions = classifier.predict(teste[:, [0, 1]])
  corretos = teste[:, [2]]
  for n, m in enumerate(predictions):
    if(m == corretos[n]):
      if(m < 0):
        irisC += 1
      else:
        versiC += 1
    else:
      if(m < 0):
        versiE += 1
      else:
        irisE += 1
  

#construir matriz de confusão 

acuracia = ((irisC + versiC)/(irisC + irisE + versiC + versiE))
print("Acurácia:", acuracia)

classes = ("Iris Setosa", "Versicolour")

print('\t')
for classe in classes:
    print("\t" + classe, end='')
print()
print(classes[0], "  ", irisC,"     ", "     ", versiE)
print(classes[1], "  ", irisE,"     ", "     ", versiC)
  #construir matriz de confusão 
