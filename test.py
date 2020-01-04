import csv
import numpy as np
from mf import MF
import pickle
import random

inTestCsv = 'ml-latest-small/testRatings.csv'
outModel = 'trainedModel.pkl'

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
		print('Read ' + str(line_count) + ' lines from ' + filename)
	print('Done reading ' + filename)
	return records

def writeCsv(records, filename):
	print('Writing ' + filename)
	with open(filename, 'w') as csv_file:
		csv_file.write('Header\n')
		for row in records:
			csv_file.write(str(row[0]) + ',' + str(row[1]) + ',' + str(row[2]) + '\n')
	print('Done writing ' + filename)

def main():
	print('Reading from ' + outModel)
	mf = 0
	with open(outModel, 'rb') as input:
		mf = pickle.load(input)
	print('Done reading from ' + outModel)

	mse = 0
	mae = 0
	testSubset = readCsv(inTestCsv)
	for row in testSubset:
		mse += pow(row[2] - mf.get_rating(row[0], row[1]), 2)
		mae += abs(row[2] - mf.get_rating(row[0], row[1]))
	mse /= len(testSubset)
	mae /= len(testSubset)

	print('[from trainSubset] mean square error = ' + str(mf.get_mse()))
	print('[from testSubstet] mean square error = ' + str(mse))
	print('[from testSubstet] mean absolute error = ' + str(mae))

	print('\nSample values:')
	print('expected','predicted')
	random.seed(0)
	for row in random.sample(testSubset, 10):
		print(row[2], mf.get_rating(row[0], row[1]))