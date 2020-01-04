import csv
import pickle
import random
import threading
import numpy as np
from mf import MF

def readCsv(filename):
	print('Reading ' + filename)
	records = []
	with open(filename) as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		line_count = 0
		for row in csv_reader:
			if line_count != 0:
				records.append([ int(row[0]), int(row[1]), float(row[2]) ])
			line_count += 1
		print('Read ' + str(line_count) + ' from ' + filename)
	print('Done reading ' + filename)
	return records

def getMaxIds(csvRecords):
	maxUserId = 0
	maxMovieId = 0
	for row in csvRecords:
		if row[0] > maxUserId:
			maxUserId = row[0]
		if row[1] > maxMovieId:
			maxMovieId = row[1]
	return maxUserId, maxMovieId
	
def getRatingsMatrix(csvRecords, maxUserId, maxMovieId):
	R = np.zeros((maxUserId+1, maxMovieId+1))
	for row in csvRecords:
		R[row[0]][row[1]] = row[2]
	return R
	
def doTrain(K, alpha, beta, gamma, iterations, maxError):
	print('>>> K=' + str(K) + ', alpha=' + str(alpha) + ', beta=' + str(beta) + ', gamma=' + str(gamma) + ', iterations=' + str(iterations) + ', maxError=' + str(maxError))
	inCsv = 'ml-latest-small/ratings.csv'
	inTestCsv = 'ml-latest-small/trainRatings.csv'
	outModel = 'trainedModel.pkl'
		
	ratings = readCsv(inCsv)
	trainSubset = readCsv(inTestCsv)
	maxUserId, maxMovieId = getMaxIds(ratings)
	R = getRatingsMatrix(trainSubset, maxUserId, maxMovieId)
	mf = MF(R, K, alpha, beta, gamma, iterations, maxError)

	print('Training...')
	training_process = mf.train()
	print('Done. Mse = ' + str(mf.get_mse()))
	
	print('Serializing model to ' + outModel)
	with open(outModel, 'wb') as output:
		pickle.dump(mf, output, pickle.HIGHEST_PROTOCOL)
	print('Done serializing model to ' + outModel)

def main():
	"""
	Models with different parameters were tested.
	From set 1, the best value for gamma is 0.06
	From set 2, the best value for K is 100
	All training and testing outputs were saved in trainedModels/outputs

	# doTrain(100, 0.06, 0.6, 0.95, 100, 0.0)
	# doTrain(100, 0.06, 0.06, 0.95, 100, 0.0) #best
	# doTrain(100, 0.06, 0.006, 0.95, 100, 0.0)
	# doTrain(100, 0.06, 0.0006, 0.95, 100, 0.0)
	# doTrain(100, 0.06, 0.00006, 0.95, 100, 0.0)
	# doTrain(100, 0.055, 0.000006, 0.95, 100, 0.0)
	# doTrain(100, 0.06, 0.0, 0.95, 100, 0.0)

	# doTrain(25, 0.06, 0.06, 0.95, 100, 0.0)
	# doTrain(50, 0.06, 0.06, 0.95, 100, 0.0)
	# doTrain(250, 0.06, 0.06, 0.95, 100, 0.0)
	# doTrain(500, 0.06, 0.06, 0.95, 100, 0.0)
	# doTrain(1000, 0.06, 0.06, 0.95, 100, 0.0)
	"""

	doTrain(100, 0.06, 0.06, 0.95, 100, 0.0) #best