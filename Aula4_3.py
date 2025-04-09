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
treino = []
for train_index, test_index in kf.split(vetores):
  for i in train_index:
    treino = treino.insert(vetores[i], -1)
    print(treino)
  for i in test_index:
    print("testar e verificar acertos")
  #construir matriz de confusão 
