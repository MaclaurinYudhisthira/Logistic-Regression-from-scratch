import numpy as np
class LogisticRegression:
    def __init__(self):
        self.iterations=100
        self.learning_rate=.01
    
    def hypotheses(self,X):
        z=self.W @ X
        return (1/(1+np.exp(-z)))

    def costFunction(self,X,Y):
        y_pred=self.hypotheses(X)
        e= 1e-5    # equivalent to 1 × 10^(−5)
        return np.sum(-Y * np.log(y_pred+e)-(1-Y)*np.log(1-y_pred+e))/self.n_sample

    
    def gardient_descent(self,X,Y):
        costs = np.zeros((self.iterations,1))

        print(f"Value of cost function before gardient descent: {self.costFunction(X,Y)}")

        for i in range(self.iterations):
            self.W=self.W-(self.learning_rate/self.n_sample)*(X @ (self.hypotheses(X) -Y))
            costs[i]=self.costFunction(X,Y)
        
        print(f"Value of cost function After {self.iterations} iterations of gardient descent: {self.costFunction(X,Y)}")

        return costs

    def fit(self,x,y,iterations=10000,learning_rate=.01):
        self.iterations=iterations
        self.learning_rate=learning_rate
        self.n_sample = len(y)

        if type(x)!=np.ndarray:
            x=np.array(x)
        X=np.hstack((np.ones((1,self.n_sample)).T,x)).T

        Y=np.array(y)

        self.W=np.zeros(X.shape[0])

        return self.gardient_descent(X,Y)

    def predict(self,x):
        if type(x)!=np.ndarray:
            x=np.array(x)
        X=np.hstack((np.ones((1,len(x))).T,x)).T
        return np.round(self.hypotheses(X))

    def params(self):
        return self.W