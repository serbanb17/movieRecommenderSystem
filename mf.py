import sys
import numpy as np
from copy import deepcopy

"""
GoodToKnow
R - matrix: rows = users, columns = movies
K - latent dimensions
alpha - learning rate
beta - regularization rate
gamma - learning acceleration rate (must be < 1; 1 means no acc)
iterations - max number of iterations
maxError - maximum error
"""

class MF():
    
	def __init__(self, R, K, alpha, beta, gamma, iterations, maxError):
		self.R = R
		self.nr_users, self.nr_events = R.shape
		self.K = K
		self.alpha = alpha
		self.beta = beta
		self.gamma = gamma
		self.iterations = iterations
		self.maxError = maxError

	def train(self):
		self.P = np.random.rand(self.nr_users, self.K)/self.K
		self.Q = np.random.rand(self.nr_events, self.K)/self.K
		self.b_u = np.zeros(self.nr_users)
		self.b_m = np.zeros(self.nr_events)
		self.b = np.mean(self.R[np.where(self.R != 0)])
		
		self.samples = [
			(i, j, self.R[i, j])
			for i in range(self.nr_users)
			for j in range(self.nr_events)
			if self.R[i, j] > 0
		]
		
		oldMse = sys.float_info.max
		for i in range(self.iterations):
			P_bak = deepcopy(self.P)
			Q_bak = deepcopy(self.Q)
			b_u_bak = deepcopy(self.b_u)
			b_m_bak = deepcopy(self.b_m)
			b_bak = deepcopy(self.b)
			np.random.shuffle(self.samples)
			self.do_sgd()
			mse = self.get_mse()
			if oldMse < mse:
				self.P = deepcopy(P_bak)
				self.Q = deepcopy(Q_bak)
				self.b_u = deepcopy(b_u_bak)
				self.b_m = deepcopy(b_m_bak)
				self.b = deepcopy(b_bak)
				self.alpha *= self.gamma
				mse = oldMse
			else:
				oldMse = mse
			print("Iteration: %d,\t error = %.10f,\t alpha=%.10f" % (i+1, mse, self.alpha))
			if mse < self.maxError:
				break;

	def get_mse(self):
		xs, ys = self.R.nonzero()
		predicted = self.get_full_matrix()
		error = 0
		for x, y in zip(xs, ys):
			error += pow(self.R[x, y] - predicted[x, y], 2)
		return error/len(xs)

	def do_sgd(self):
		for i, j, r in self.samples:
			prediction = self.get_rating(i, j)
			e = (r - prediction)
			
			self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i])
			self.b_m[j] += self.alpha * (e - self.beta * self.b_m[j])
			
			P_i = self.P[i, :][:]
			
			self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta * self.P[i,:])
			self.Q[j, :] += self.alpha * (e * P_i - self.beta * self.Q[j,:])

	def get_rating(self, i, j):
		prediction = self.b + self.b_u[i] + self.b_m[j] + self.P[i, :].dot(self.Q[j, :].T)
		return prediction
	
	def get_full_matrix(self):
		return self.b + self.b_u[:,np.newaxis] + self.b_m[np.newaxis:,] + self.P.dot(self.Q.T)
