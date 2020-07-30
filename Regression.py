import numpy as np 

class Regression:

	def __init__(self,order = 1):

		self.Cost = []
		self.order = order

	def poly(self,X,y,order):

		m = X.shape[0]
		# Adding the polynomial
		X = np.hstack((
	    X,
	    (X[:, 0] ** order).reshape((m, 1))/np.argmax(X[:,0]),
		))
		# Normalization
		X = (X - np.mean(X))/np.std(X)

		StdInput = np.std(X)
		MeanInput = np.mean(X)

		NormalizationParameters = {'StdInput':StdInput, 'MeanInput':MeanInput}

		return X, NormalizationParameters

	def initialize(self, X, y):

		W = np.zeros((1,X.T.shape[0]))
		b = 0
		return W,b 

	def cost(self, X, y, W, b):

		m = X.T.shape[1]
		cost = (1/(2*m)) * np.sum(((np.dot(W,X.T)+b) - y.T)**2)
		return cost

	def fit(self,X,y,iterations=400,learning_rate=0.1, lamd = 1, order=1.5, costoutput=True,showfittedline=True ,Optimizer = 'GradientDescent', beta=0.99, epsilon=10**(-8), beta1=0.9, beta2=0.999):

		X, NormalizationParameters = self.poly(X, y, order)
		self.W,self.b = self.initialize(X, y)
		m = X.T.shape[1]

		dW = (1/m) * (np.sum(np.dot(((np.dot(self.W,X.T)+self.b) - y.T),X), axis=0) + np.sum(lamd * self.W))
		db = (1/m) * (np.sum((np.dot(self.W,X.T)+self.b) - y.T , axis=-1) + np.sum(lamd * self.b))

		# Gradient Descent Optimizer
		if Optimizer == 'GradientDescent':
			for i in range(iterations):

				self.W = self.W - learning_rate * dW
				self.b = self.b - learning_rate * db

				self.Cost.append(self.cost(X,y,self.W,self.b))

		# Momentum Optimizer
		elif Optimizer == 'Momentum':
			VdW = 0
			Vdb = 0
			for i in range(iterations):
				VdW = beta*VdW + dW
				Vdb = beta*Vdb + db

				self.W = self.W - learning_rate * (1/m) * VdW
				self.b = self.b - learning_rate * (1/m) * Vdb

				self.Cost.append(self.cost(X,y,self.W,self.b))

		# RMSprop Optimizer
		elif Optimizer == 'RMSprop':
			SdW = 0
			Sdb = 0
			for i in range(iterations):
				SdW = beta * SdW + (1 - beta) * dW**2
				Sdb = beta * Sdb + (1 - beta) * db**2

				self.W = self.W - learning_rate * (dW/(SdW+epsilon)**(0.5))
				self.b = self.b - learning_rate * (db/(Sdb+epsilon)**(0.5))

				self.Cost.append(self.cost(X,y,self.W,self.b))

		# Adam Optimizer
		elif Optimizer == 'Adam':
			VdW = 0
			Vdb = 0
			SdW = 0
			Sdb = 0
			for i in range(iterations):
				VdW = beta1*VdW +(1 - beta1) * dW
				Vdb = beta1*Vdb +(1 - beta1) * db

				SdW = beta2 * SdW + (1 - beta2) * dW**2
				Sdb = beta2 * Sdb + (1 - beta2) * db**2

				# print(SdW)

				VdW_correct = VdW/(1 - beta1**(iterations))
				Vdb_correct = Vdb/(1 - beta1**(iterations))

				self.W = self.W - learning_rate * (SdW/(VdW_correct+epsilon))
				self.b = self.b - learning_rate * (Sdb/(Vdb_correct+epsilon))

				self.Cost.append(self.cost(X,y,self.W,self.b))

		if costoutput == True:
			line = np.dot(self.W,X.T)+self.b
			import matplotlib.pyplot as plt
			# Will give the fitted line 
			if showfittedline == True:
				plt.plot(X[:,0],line.T,'+')
				plt.plot(X[:,0],y,'+')
				plt.show()
			# Will give the cost
			plt.plot(self.Cost,label=f'learning rate = {learning_rate}')
			plt.xlabel('#iterations')
			plt.ylabel(f'Cost')
			plt.legend()
			plt.show()

		param = {'W':self.W, 'b':self.b}
		return param, NormalizationParameters

	def predict(self,X,y,param,NormalizationParameters):
		W,b = param['W'],param['b']
		StdInput,MeanInput = NormalizationParameters['StdInput'],NormalizationParameters['MeanInput']
		try:
			m = X.shape[0]
			X = np.hstack((
		    X,
		    (X[:, 0] ** self.order).reshape((m, 1))/np.argmax(X[:,0]),
			))
			y = np.dot(W, X.T) + b
			y = (y*StdInput+MeanInput)/1000
			return y 
		except Exception as e:
			print(e)
