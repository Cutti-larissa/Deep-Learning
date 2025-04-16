#Tá certo??
negativo = np.random.uniform(1.2,0.7,(10,2))
positivo = np.array(negativo)

negativo[0:5, :] *= -1
positivo[0:5,0] *= -1
positivo[5:10,1] *= -1

#essas serão as entradas da sua rede
inputs = np.concatenate([negativo, positivo])

#primeiro perceptron já feito, como exemplo
#primeiro perceptron
pp = Perceptron()
pp.n_features_in_ = 2
pp.classes_ = np.array([-1, 1])
#intercept é o bias
pp.intercept_ = np.array([0.5])
#vetor com os demais pesos w0 e w1
pp.coef_ = np.array([[1, 1]])

#segundo perceptron
sp = Perceptron()
sp.n_features_in_ = 2
sp.classes_ = np.array([-1, 1])
#intercept é o bias
sp.intercept_ = np.array([-1.5])
#vetor com os demais pesos w0 e w1
sp.coef_ = np.array([[1, 1]])

#para gerar a predição de todos os vetores, a partir de um perceptron, use predict. Exemplo:
r1 = pp.predict(inputs)
r2 = sp.predict(inputs)

#se precisar combinar as saídas de dois perceptrons em duas colunas, use a função stack. Exemplo:
#np.column_stack((r1,r2))
input2 = np.column_stack((r1,r2))

#terceiro perceptron
tp = Perceptron()
tp.n_features_in_ = 2
tp.classes_ = np.array([-1, 1])
#intercept é o bias
tp.intercept_ = np.array([-1])
#vetor com os demais pesos w0 e w1
tp.coef_ = np.array([[0.7, -1.4]])

outputs = tp.predict(input2)

#salve as respostas em um vetor chamado outputs e descomente o código abaixo para
#exibir os resultados em um gráfico

plt.xlabel("x1")
plt.ylabel("x2")
plt.axvline(x=0, c="black", label="x=0")
plt.axhline(y=0, c="black", label="y=0")
plt.xlim(-2, 2)
plt.ylim(-2, 2)

for i, input in enumerate(inputs):
  if outputs[i] >= 0:
    plt.plot(input[0], input[1], '+', color='blue')
  else:
    plt.plot(input[0], input[1], 'o', color='red')
