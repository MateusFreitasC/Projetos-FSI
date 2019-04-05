from __future__ import print_function
import sys
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import math
from skimage import exposure
import numpy as np
import imutils
import cv2 as cv
import sklearn

if int((sklearn.__version__).split(".")[1]) < 18:
	from sklearn.cross_validation import train_test_split
 
# otherwise we're using at lease version 0.18
else:
	from sklearn.model_selection import train_test_split

# Redimensiona a imagem --------------------------------------------------
def resize(mult, oriImg):
	height, width = oriImg.shape
	imgScale = mult/width
	newX,newY = oriImg.shape[1]*imgScale, oriImg.shape[0]*imgScale
	return cv.resize(oriImg,(int(newX),int(newY)))

# Retorna as imagens dos quadrantes e o número de pixels brancos de cada quadrante inferior do menor retângulo ------------------------------------
def nonZeroPerQuad(imgOri, x, y, w, h):
	halfL = imgOri[y: y+h,x: x+(w//2)]
	halfR = imgOri[y: y+h,x+(w//2): x+w]
	halfUp = imgOri[y:y+(h//2), x:x+w]
	halfDown = imgOri[y+(h//2):y+h, x:x+w]
	quad1 = imgOri[y:y+(h//2), x+(w//2):x+w]
	quad2 = imgOri[y:y+(h//2):, x:x+(w//2)]
	quad3 = imgOri[y+(h//2):y+h, x:x+(w//2)]
	quad4 = imgOri[y+(h//2):y+h, x+(w//2):x+w]
	
	return 	cv.countNonZero(quad1), cv.countNonZero(quad2), 		cv.countNonZero(quad3), cv.countNonZero(quad4)

# Retorna a imagem cortada e rotacionada para o menor retângulo ----------
def imgRect(img):
	ret,img = cv.threshold(img, 127, 255, 0)	
	contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
	rect = cv.minAreaRect(contours[0])
	rectPoints = cv.boxPoints(rect)
	y1 = rectPoints[0][1]
	y2 = rectPoints[1][1]
	y3 = rectPoints[2][1]
	y4 = rectPoints[3][1]
	center, (width, height), theta = rect
	
	shape = (img.shape[1], img.shape[0])
	# cv2.warpAffine expects shape in (length, height)
	
	if y2 < y1 and y2 > y4:
		theta += 90

	matrix = cv.getRotationMatrix2D(center=center, angle=theta, scale=1)
	img = cv.warpAffine(src=img, M=matrix, dsize=shape)
	
	ret,thresh = cv.threshold(img, 127, 255, 0)
	contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
	x,y,w,h = cv.boundingRect(contours[0])

	return thresh, (x, y), (w, h)

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
	newTrainData = []
	newTrainLabels = []
	for img, label in zip(trainData, trainLabels):
		thresh, (x, y), (w, h) = imgRect(img)
		contours,hierarchy = cv.findContours(thresh, 1, 2)
		areaInt = cv.contourArea(contours[0])
		whitePixels = cv.countNonZero(thresh)
		p1, p2, p3, p4 = nonZeroPerQuad(thresh, x, y, w, h)
					
		newTrainData.append([whitePixels, (p2+p3)/(p1+p4), (p1+p2)/(p3+p4), p1, p2, p3, p4, w*h, areaInt])
		newTrainLabels.append(label)
		
	# Montando novos atributos para os testes
	newTestData = []
	newTestLabels = []
	for img, label in zip(testData, testLabels):
		thresh, (x, y), (w, h) = imgRect(img)
		contours,hierarchy = cv.findContours(thresh, 1, 2)
		areaInt = cv.contourArea(contours[0])
		whitePixels = cv.countNonZero(thresh)
		p1, p2, p3, p4 = nonZeroPerQuad(thresh, x, y, w, h)

		newTestData.append([whitePixels, (p2+p3)/(p1+p4), (p1+p2)/(p3+p4), p1, p2, p3, p4, w*h, areaInt])
		newTestLabels.append(label)

	newTrainData = np.array(newTrainData)
	newTrainLabels = np.array(newTrainLabels)
	newTestData = np.array(newTestData)
	newTestLabels = np.array(newTestLabels)

	# Separando os dados de treino (90% -> treino e 10% -> validação)
	(newTrainData, valData, newTrainLabels, valLabels) = train_test_split(newTrainData, newTrainLabels, test_size=0.1, random_state=84)
	 
	# Mostrar os dados separados
	print("Dados para treino: {}".format(len(newTrainLabels)))
	print("Dados para validação: {}".format(len(valLabels)))
	print("Dados para teste: {}".format(len(newTestLabels)))

	# Pega os valores ímpares para k entre 1 e 11
	kVals = range(1, 11, 2)
	accuracies = []
 
	# Para cada valor de K executa KNN para o conjunto de dados adquiridos
	print(">> Iniciando treinamento ...")
	for k in kVals:
		# Treina o classificador para o k definido
		print("> Treinamento para K = " + str(k), end='')
		model = KNeighborsClassifier(n_neighbors=k)
		model.fit(newTrainData, newTrainLabels)
	 
		# Pega o resultado do treinamento e armazena em 'accuracies'
		score = model.score(valData, valLabels)
		print(" -> PRECISÃO = %.2f%%" % (score * 100))
		accuracies.append(score)
	 
	# Encontra o valor de k com a maior eficiência
	i = int(np.argmax(accuracies))
	print("K = %d adquiriu a maior precisão (%.2f%%) para os dados de validação fornecidos" % (kVals[i], accuracies[i] * 100))
	
	print("> Efetuando os testes para o melhor k (k=" + str(kVals[i]) + ")")
	model = KNeighborsClassifier(n_neighbors=kVals[i])
	model.fit(newTrainData, newTrainLabels)
	predictions = model.predict(newTestData)
 
	# Mostra os resultados dos testes para cada dígito
	print("RESULTADO DOS TESTES PARA O MELHOR K")
	print(classification_report(newTestLabels, predictions))

	for i in list(map(int, np.random.randint(0, high=len(newTestLabels), size=(10,)))):
		# Pega uma imagem aleatória dos dados de teste
		image = testData[i]
		prediction = model.predict(newTestData[i].reshape((1,-1)))[0]

		ret,thresh = cv.threshold(resize(500, image), 127, 255, 0)	
		contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
		rect = cv.minAreaRect(contours[0])
		rectPoints = cv.boxPoints(rect)
		cv.drawContours(thresh,[np.int0(rectPoints)],0,(255,255,255),1)

		cv.imshow('img Test', thresh)
		cv.waitKey(0)
		
		threshRE, (xRE, yRE), (wRE, hRE) = imgRect(resize(500, image))
		threshRGB = cv.cvtColor(threshRE, cv.COLOR_GRAY2RGB)

		cv.rectangle(threshRGB, (xRE+(wRE//2),yRE), (xRE+wRE,yRE+(hRE//2)), (0,0,255), 1)	
		cv.rectangle(threshRGB, (xRE,yRE), (xRE+(wRE//2),yRE+(hRE//2)), (0,255,0), 1)	
		cv.rectangle(threshRGB, (xRE,yRE+(hRE//2)), (xRE+(wRE//2),yRE+hRE), (255,0,0), 1)	
		cv.rectangle(threshRGB, (xRE+(wRE//2),yRE+(hRE//2)), (xRE+wRE,yRE+hRE), (255,0,255), 1)	

		cv.putText(threshRGB,'pix1Q = %d'%(newTestData[i][3]), (160,20), cv.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 255), 1)
		cv.putText(threshRGB,'pix2Q = %d'%(newTestData[i][4]), (10,20), cv.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 0), 1)
		cv.putText(threshRGB,'pix3Q = %d'%(newTestData[i][5]), (10,40), cv.FONT_HERSHEY_SIMPLEX, .7, (255, 0, 0), 1)
		cv.putText(threshRGB,'pix4Q = %d'%(newTestData[i][6]), (160,40), cv.FONT_HERSHEY_SIMPLEX, .7, (255, 0, 255), 1)
		cv.putText(threshRGB,'hL/hR = %.2f'%(newTestData[i][1]), (10,65), cv.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 1)		
		cv.putText(threshRGB,'hU/hD = %.2f'%(newTestData[i][2]), (180,65), cv.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 1)

		cv.putText(threshRGB, "{} ?".format(prediction), (340, 35), cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

		if prediction == testLabels[i]:
			cv.putText(threshRGB, 'Y', (410, 35), cv.FONT_ITALIC, 1, (0,255,0), 2)
		else:
			cv.putText(threshRGB, 'N', (410, 35), cv.FONT_ITALIC, 1, (0,0,255), 2)
	 
		# Mostra a previsão para a imagem escolhida
		print("I think that digit is: {}".format(prediction))
		cv.imshow("Image (" + str(testLabels[i]) + ")", threshRGB)
		cv.waitKey(0)
		cv.destroyAllWindows()

if __name__ == '__main__':
	main(sys.argv[1:])


