import urllib.request
from pathlib import Path
import zipfile
import splitSubsets
import train
import test

urlToMovieLensDataset = 'http://files.grouplens.org/datasets/movielens/ml-latest-small.zip'
fileName = 'ml-latest-small.zip'

if Path(fileName).is_file():
	print(fileName + ' exists. No need to download')
else:
	print('Downloading ' + fileName + ' from ' + urlToMovieLensDataset)
	urllib.request.urlretrieve(urlToMovieLensDataset, fileName)

print('Unzipping ' + fileName);	
with zipfile.ZipFile(fileName, 'r') as zip_ref:
	zip_ref.extractall('')

print('\n\n\n\n\n')
print('Running splitSubsets.py: it will separate ml-latest-small/ratings.csv into training and test subsets')
splitSubsets.main()

print('\n\n\n\n\n')
print('Running train.py')
train.main()

print('\n\n\n\n\n')
print('Running test.py')
test.main()