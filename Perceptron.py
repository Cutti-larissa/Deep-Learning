incorretos = vetores
#Implemente aqui o algoritmo de treinamento
#Enquanto houverem instâncias rotuladas incorretamente:
#while(len(incorretos) != 0):
for x in vetores: #Passe por cada instância x de treinamento em M.
  x_bias = x
  x_bias = np.delete(x_bias, -1, 0)
  prod_escalar = w.dot(x_bias)
  print(prod_escalar)
  if (((prod_escalar < 0) & x[-1] < 0) |((prod_escalar > 0) & x[-1] > 0)):
    #remove x de incorretos #Se x foi classificada corretamente, não faça nada.
  else: #Se x foi classificada incorretamente, faça:
    #w = w + eta(xy)