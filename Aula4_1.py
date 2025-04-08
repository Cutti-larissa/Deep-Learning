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
vetores_classe_0 = vetores_classe_0.drop(columns = remover)
vetores_classe_1 = vetores_classe_1.drop(columns = remover)

#ajustando classes para operar com o perceptron
vetores_classe_0["target"] = - 1
vetores_classe_1["target"] = + 1

#colocando em um vetor numpy para facilitar
vetores = np.concatenate((vetores_classe_0.to_numpy(), vetores_classe_1.to_numpy()))

#usando a funçao pronta do scikit-leanr para criar os conjuntos de treinamento e teste
#50% para treinamento, 50% para teste
train, test = train_test_split(vetores, test_size=0.5)

#Agora que já sabemos criar um perceptron, vamos usar os perceptron pronto do Scikit-Learn
classifier = Perceptron(eta0 = 0.05)
classifier.fit(train[:,[0,1]], train[:,2])

#passa por cada um dos vetores de testes, e guarda a predição do modelo
predictions = classifier.predict(test[:, [0, 1]])

contar = 0 
#crie aqui um loop que compara as predições do modelo com as classes reais do teste (ground-truth)
for i in range(len(predictions)):
  if (predictions[i] == test[i][-1]):
    contar += 1
#com isso, gere a acurácia do modelo
acuracia = contar / len(test)
print(acuracia)
#depois, compare seu resultado com a função pronta do scikit-learn accuracy = accuracy_score(test[:, 2], predictions), que executa exatamente esse algoritmo
accuracy = accuracy_score(test[:, 2], predictions)
print(accuracy)
