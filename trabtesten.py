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

# Redimensiona a imagem --------------------------------------------------
def resize(mult, oriImg):
	height, width = oriImg.shape
	imgScale = mult/width
	newX,newY = oriImg.shape[1]*imgScale, oriImg.shape[0]*imgScale
	return cv.resize(oriImg,(int(newX),int(newY)))

# Retorna um vetor com a quantidade de pixels brancos de cada quadrante definido pelo valor de entrada (3º argumento do terminal)
def MontaDados(img, div, model):
	ret, img = cv.threshold(img, 127, 255, 0)	
	contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
	rect = cv.minAreaRect(contours[0])
	rectPoints = cv.boxPoints(rect)
	y1 = rectPoints[0][1]
	y2 = rectPoints[1][1]
	y3 = rectPoints[2][1]
	y4 = rectPoints[3][1]
	center, (width, height), theta = rect
	
	shape = (img.shape[1], img.shape[0])
	
	if y2 < y1 and y2 > y4:
		theta += 90

	matrix = cv.getRotationMatrix2D(center=center, angle=theta, scale=1)
	img = cv.warpAffine(src=img, M=matrix, dsize=shape)
	contours,hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
	x, y, w, h = cv.boundingRect(contours[0])	
		
	if model == 'knn': 
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

	elif model == 'lda':		
		dataLDAx = cv.countNonZero(img[y:int(y+h), x:int(x+w)])		
		dataLDAy = cv.contourArea(contours[0])

		return dataLDAx, dataLDAy
	
	else: return None

# Main -------------------------------------------------------------------
def main(argv):
	print('> Coletando os dados MNIST para iniciar procedimentos de teste ...')
	# Carregando os dados MNIST
	(trainData, trainLabels), (testData, testLabels) = tf.keras.datasets.mnist.load_data()
	
	# Dividindo os dados de treino e teste para valores desejados
	trainData = np.split(trainData, int(sys.argv[1]))[0]
	trainLabels = np.split(trainLabels, int(sys.argv[1]))[0]
	testData = np.split(testData, int(sys.argv[2]))[0]
	testLabels = np.split(testLabels, int(sys.argv[2]))[0]
	 
	# Montando novos atributos para o treino
	newTrainDataKnn = []
	newTrainDataLda = [[], [], [], [], [], [], [], [], [], [],
					   [], [], [], [], [], [], [], [], [], []]
	newTrainLabels = []
	i = 1
	for img, label in zip(trainData, trainLabels):
		print('> Montando trainData[%d/%d]'%(i, len(trainData)))		
		newTrainDataKnn.append(MontaDados(img, int(sys.argv[3]), 'knn'))
		trainLDAx, trainLDAy = MontaDados(img, int(sys.argv[3]), 'lda')
		newTrainDataLda[label].append(trainLDAx)
		newTrainDataLda[label+10].append(trainLDAy)
		newTrainLabels.append(label)
		i += 1
		
	# Montando novos atributos para os testes
	newTestDataKnn = []
	newTestDataLda = [[], [], [], [], [], [], [], [], [], [],
					  [], [], [], [], [], [], [], [], [], []]
	newTestLabels = []
	i = 1
	for img, label in zip(testData, testLabels):
		print('> Montando testData[%d/%d]'%(i, len(testData)))
		newTestDataKnn.append(MontaDados(img, int(sys.argv[3]), 'knn'))
		testLDAx, testLDAy = MontaDados(img, int(sys.argv[3]), 'lda')
		newTrainDataLda[label].append(trainLDAx)
		newTrainDataLda[label+10].append(trainLDAy)
		newTestLabels.append(label)
		i += 1

	newTrainDataKnn = np.array(newTrainDataKnn)
	newTrainDataLda = np.array(newTrainDataLda)
	newTrainLabels = np.array(newTrainLabels)
	newTestDataKnn = np.array(newTestDataKnn)
	newTestDataLda = np.array(newTestDataLda)
	newTestLabels = np.array(newTestLabels)
	
	# Separando os dados de treino (90% -> treino e 10% -> validação)
	(newTrainDataKnn, valData, newTrainLabels, valLabels) = train_test_split(newTrainDataKnn, newTrainLabels, test_size=0.1, random_state=84)
	 
	# Mostrar os dados separados
	print("\nDados para treino: {}".format(len(newTrainLabels)))
	print("Dados para validação: {}".format(len(valLabels)))
	print("Dados para teste: {}\n".format(len(newTestLabels)))

	# Pega os valores ímpares para k entre 1 e 11
	kVals = range(1, 11, 2)
	accuracies = []
 
	# Para cada valor de K executa KNN para o conjunto de dados adquiridos
	print(">> Iniciando treinamento [%d quadrantes] ..."%(int(sys.argv[3])))
	for k in kVals:
		# Treina o classificador para o k definido
		print("> Treinamento para K = " + str(k), end='')
		modelKNN = KNeighborsClassifier(n_neighbors=k)
		modelKNN.fit(newTrainDataKnn, newTrainLabels)
	 
		# Pega o resultado do treinamento e armazena em 'accuracies'
		score = modelKNN.score(valData, valLabels)
		print(" -> PRECISÃO = %.2f%%" % (score * 100))
		accuracies.append(score)
	 
	# Encontra o valor de k com a maior eficiência
	i = int(np.argmax(accuracies))
	print("\nK = %d adquiriu a maior precisão (%.2f%%) para os dados de validação fornecidos" % (kVals[i], accuracies[i] * 100))
	
	print("\n> Efetuando os testes para o melhor k (k=" + str(kVals[i]) + ")\n")
	modelKNN = KNeighborsClassifier(n_neighbors=kVals[i])
	modelKNN.fit(newTrainDataKnn, newTrainLabels)
	predictionsKNN = modelKNN.predict(newTestDataKnn)
 
	# Mostra os resultados dos testes Knn
	print("     >>> RESULTADO DOS TESTES PARA O MELHOR K - MODELO KNN <<<\n")
	print(classification_report(newTestLabels, predictionsKNN))

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

	fig = plt.figure(figsize=(10,10))
	ax0 = fig.add_subplot(111)
	ax0.scatter(newTrainDataLda[0], newTrainDataLda[10], marker='s', c='grey', edgecolor='black')
	ax0.scatter(newTrainDataLda[1], newTrainDataLda[11], marker='s', c='blue', edgecolor='black')
	ax0.scatter(newTrainDataLda[2], newTrainDataLda[12], marker='s', c='green', edgecolor='black')
	ax0.scatter(newTrainDataLda[3], newTrainDataLda[13], marker='s', c='pink', edgecolor='black')
	ax0.scatter(newTrainDataLda[4], newTrainDataLda[14], marker='s', c='red', edgecolor='black')
	ax0.scatter(newTrainDataLda[5], newTrainDataLda[15], marker='s', c='yellow', edgecolor='black')
	ax0.scatter(newTrainDataLda[6], newTrainDataLda[16], marker='s', c='purple', edgecolor='black')
	ax0.scatter(newTrainDataLda[7], newTrainDataLda[17], marker='s', c='black', edgecolor='black')
	ax0.scatter(newTrainDataLda[8], newTrainDataLda[18], marker='s', c='orange', edgecolor='black')
	ax0.scatter(newTrainDataLda[9], newTrainDataLda[19], marker='s', c='white', edgecolor='black')
	#mean_data = np.mean(newTrainDataLda, axis=1).reshape(2,1)
	#scatter_data = np.dot((newTrainDataLda-mean_data),(newTrainDataLda-mean_data).T)
	plt.show()

if __name__ == '__main__':
	main(sys.argv[1:])
