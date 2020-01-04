import csv, random

splitRatio = 0.9
randSeed = 9
inCsv = 'ml-latest-small/ratings.csv'
outTrainCsv = 'ml-latest-small/trainRatings.csv'
outTestCsv = 'ml-latest-small/testRatings.csv'

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

def writeCsv(records, filename):
	print('Writing ' + filename)
	with open(filename, 'w') as csv_file:
		csv_file.write('Header\n')
		for row in records:
			csv_file.write(str(row[0]) + ',' + str(row[1]) + ',' + str(row[2]) + '\n')
	print('Done writing ' + filename)

def main():
	ratings = readCsv(inCsv)

	print('Preparing test & train data...')
	random.seed(a=randSeed)
	trainSubset, testSubset = [], []
	for row in ratings:
		if random.uniform(0,1) <= splitRatio:
			trainSubset.append(row)
		else:
			testSubset.append(row)
	print('Done preparing test & train data...')

	writeCsv(trainSubset, outTrainCsv)
	writeCsv(testSubset, outTestCsv)