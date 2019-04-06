from __future__ import print_function
import sys
import tensorflow as tf
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import math
from skimage import exposure
import numpy as np
import imutils
import cv2 as cv
import sklearn
import time
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Retorna um vetor com a área interna no número e a quantidade de pixels brancos de cada quadrante definido pelo valor de entrada (3º argumento na linha de comando) se model == 'knn'
# TESTE - Retorna a quantidade de pixels da imagem e a área interna do número se model == 'lda' 
def MontaDados(img, div, model):
	
	ret, img = cv.threshold(img, 127, 255, 0)	
	contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
	rect = cv.minAreaRect(contours[0])
	rectPoints = cv.boxPoints(rect) # Vértices do menor retângulo \/
	y1 = rectPoints[0][1]
	y2 = rectPoints[1][1]
	y3 = rectPoints[2][1]
	y4 = rectPoints[3][1]
	center, (width, height), theta = rect
	
	shape = (img.shape[1], img.shape[0])
	
	if y2 < y1 and y2 > y4:
		theta += 90

	matrix = cv.getRotationMatrix2D(center=center, angle=theta, scale=1)
	img = cv.warpAffine(src=img, M=matrix, dsize=shape) # Rotaciona imagem
	contours,hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
	x, y, w, h = cv.boundingRect(contours[0])	# Pega novamente as medidas do retângulo rotacionado  / x e y coordenadas da origem do retângulo 
		
	if model == 'knn':  # Retorna os dados para o modelo knn - lista de tamanho div² + 1
		dataKnn = []
		dataKnn.append(cv.contourArea(contours[0]))
		w = w/div
		h = h/div

		for i in range(div):
			for j in range(div):
				dataKnn.append(cv.countNonZero(img[int(y):int(y+h), int(x):int(x+w)]))
				x += w
		
			y += h
			x -= div*w
		
		return dataKnn

	elif model == 'lda': # TESTE - Retorna 2 dados para o modelo lda
		dataLDAx = cv.countNonZero(img[y:int(y+h), x:int(x+w)])		
		dataLDAy = cv.contourArea(contours[0])

		return dataLDAx, dataLDAy
	
	else: return None

# Main -------------------------------------------------------------------
def main(argv):
	print('> Coletando os dados MNIST para iniciar procedimentos de teste ...')
	# Carregando os dados MNIST
	(trainData, trainLabels), (testData, testLabels) = tf.keras.datasets.mnist.load_data()
	
	# Dividindo os dados de treino e teste para o valor desejado de entrada (3º argumento da linha de comando)
	trainData = np.split(trainData, int(sys.argv[1]))[0]
	trainLabels = np.split(trainLabels, int(sys.argv[1]))[0]
	testData = np.split(testData, int(sys.argv[2]))[0]
	testLabels = np.split(testLabels, int(sys.argv[2]))[0]
	 
	# Montando novos atributos para o treino knn e lda usando os dados coletados
	newTrainDataKnn = []
	newTrainDataLda = [[], [], [], [], [], [], [], [], [], [],
					   [], [], [], [], [], [], [], [], [], []]
	i = 1
	for img, label in zip(trainData, trainLabels):
		print('> Montando trainData[%d/%d]'%(i, len(trainData)))		
		newTrainDataKnn.append(MontaDados(img, int(sys.argv[3]), 'knn'))
		trainLDAx, trainLDAy = MontaDados(img, int(sys.argv[3]), 'lda')
		newTrainDataLda[label].append(trainLDAx)
		newTrainDataLda[label+10].append(trainLDAy)
		i += 1
		
	# Montando novos atributos para os testes
	newTestDataKnn = []
	newTestDataLda = [[], [], [], [], [], [], [], [], [], [],
					  [], [], [], [], [], [], [], [], [], []]
	i = 1
	for img, label in zip(testData, testLabels):
		print('> Montando testData[%d/%d]'%(i, len(testData)))
		newTestDataKnn.append(MontaDados(img, int(sys.argv[3]), 'knn'))
		testLDAx, testLDAy = MontaDados(img, int(sys.argv[3]), 'lda')
		newTrainDataLda[label].append(trainLDAx)
		newTrainDataLda[label+10].append(trainLDAy)
		i += 1

	# Padronizando as novas listas com Numpy
	newTrainDataKnn = np.array(newTrainDataKnn)
	newTrainDataLda = np.array(newTrainDataLda)
	newTestDataKnn = np.array(newTestDataKnn)
	newTestDataLda = np.array(newTestDataLda)
	
	# Separando os dados de treino (90% -> treino e 10% -> validação)
	(newTrainDataKnn, valData, newTrainLabels, valLabels) = train_test_split(newTrainDataKnn, trainLabels, test_size=0.1, random_state=84)
	 
	# Mostrar os dados separados
	print("\nDados para treino: {}".format(len(trainLabels)))
	print("Dados para validação: {}".format(len(valLabels)))
	print("Dados para teste: {}\n".format(len(testLabels)))

	# Pega os valores ímpares para k entre 1 e 11
	kVals = range(1, 11, 2)
	accuracies = []
 
	# Para cada valor de K executa KNN para o conjunto de dados adquiridos
	print(">> Iniciando treinamento [%d quadrantes] ..."%(int(sys.argv[3])))
	for k in kVals:
		# Treina o classificador para o k definido
		print("> Treinamento para K = " + str(k), end='')
		modelKNN = KNeighborsClassifier(n_neighbors=k) # Instância do modelo knn
		modelKNN.fit(newTrainDataKnn, trainLabels) # Inserindo dados de treino
	 
		# Pega o resultado do treinamento e armazena em 'accuracies'
		score = modelKNN.score(valData, valLabels) # Resultado da validação para o k estudado
		print(" -> PRECISÃO = %.2f%%" % (score * 100))
		accuracies.append(score) # Guarda o resultado em accuracies
	 
	# Encontra o valor de k com a maior eficiência
	i = int(np.argmax(accuracies)) # Índice do melhor k
	print("\nK = %d adquiriu a maior precisão (%.2f%%) para os dados de validação fornecidos" % (kVals[i], accuracies[i] * 100))
	
	print("\n> Efetuando os testes para o melhor k (k=" + str(kVals[i]) + ")\n")
	modelKNN = KNeighborsClassifier(n_neighbors=kVals[i]) # Instância do modelo knn para o melhor valor de k
	modelKNN.fit(newTrainDataKnn, trainLabels) # Inserindo novamente dados de treino
	predictionsKNN = modelKNN.predict(newTestDataKnn) # Resultado das tentativas de predição do knn para o melhor k
 
	# Mostra os resultados dos testes Knn
	print("     >>> RESULTADO DOS TESTES PARA O MELHOR K - MODELO KNN <<<\n")
	print(classification_report(testLabels, predictionsKNN)) # Mostra o classification report do resultado do knn para o melhor k por meio de comparação com o testLabels

	# TESTE - Tentativa de implementação do modelo lda (provavelmente errado)
	modelLDA = LinearDiscriminantAnalysis(n_components = 2)
	X_train = modelLDA.fit_transform(newTrainDataKnn, newTrainLabels)
	X_test = modelLDA.transform(newTestDataKnn)
	print(X_train)
	print(X_test)
	modelLDA.fit(X_train, newTrainLabels)
	predictionsLDA = modelLDA.predict(X_test)

	# Mostra os resultados dos testes LDA
	print("\t  >>> RESULTADO DOS TESTES PARA O MODELO LDA <<<\n")
	print(classification_report(newTestLabels, predictionsLDA))

	# Plotando em um plano xy os (visualização do usuário) pontos referentes aos dados MNIST importados
	fig = plt.figure(figsize=(10,10))
	ax0 = fig.add_subplot(111)
	ax0.scatter(newTrainDataLda[0], newTrainDataLda[10], marker='s', c='grey', edgecolor='black') # Fotos dos números 0's
	ax0.scatter(newTrainDataLda[1], newTrainDataLda[11], marker='s', c='blue', edgecolor='black') # Fotos dos números 1's
	ax0.scatter(newTrainDataLda[2], newTrainDataLda[12], marker='s', c='green', edgecolor='black') # Fotos dos números 2's
	ax0.scatter(newTrainDataLda[3], newTrainDataLda[13], marker='s', c='pink', edgecolor='black') # Fotos dos números 3's
	ax0.scatter(newTrainDataLda[4], newTrainDataLda[14], marker='s', c='red', edgecolor='black') # Fotos dos números 4's
	ax0.scatter(newTrainDataLda[5], newTrainDataLda[15], marker='s', c='yellow', edgecolor='black') # Fotos dos números 5's
	ax0.scatter(newTrainDataLda[6], newTrainDataLda[16], marker='s', c='purple', edgecolor='black') # Fotos dos números 6's
	ax0.scatter(newTrainDataLda[7], newTrainDataLda[17], marker='s', c='black', edgecolor='black') # Fotos dos números 7's
	ax0.scatter(newTrainDataLda[8], newTrainDataLda[18], marker='s', c='orange', edgecolor='black') # Fotos dos números 8's
	ax0.scatter(newTrainDataLda[9], newTrainDataLda[19], marker='s', c='white', edgecolor='black') # Fotos dos números 9's
	plt.show()

if __name__ == '__main__':
	main(sys.argv[1:])
