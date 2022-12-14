## implements a random features model using sklearn. Combines data preprocessing and function fitting into a single estimator using a pipeline. 
import numpy as np
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import FunctionTransformer,StandardScaler
from sklearn.pipeline import Pipeline

def get_random_linear_projector(matrix):
    """defines a function that multiples by a given random matrix, and creates a sklearn transformer that applies this function. 

    """
    def projector(data):
        """Assumes data is of shape (batch,features)

        """
        return np.matmul(data,matrix.T)
    return FunctionTransformer(projector)

def get_random_fourier_projector(matrix,offset):
    """defines a function that multiples by a given random matrix, gets a random fourier feature, and creates a sklearn transformer that applies this function. 

    """
    def projector(data):
        """Assumes data is of shape (batch,features)

        """
        return np.cos(np.matmul(data,matrix.T)+offset)
    return FunctionTransformer(projector)
    

def get_linear_pipelined_logistic_classification(in_d,out_d,sigma=1,logistic_args={},random_state=None):    
    """returns a pipeline that combines standard scaling, projection with a random matrix, and logistic classification into a single pipelined estimator that can then be bagged.  

    """
    matrix =  sklearn.utils.check_random_state(random_state).randn(out_d,in_d)/np.sqrt(sigma*in_d)

    full_estimator = Pipeline([('standardscaler',StandardScaler()),('projector',get_random_linear_projector(matrix)),('logistic',LogisticRegression(**logistic_args))])
    return full_estimator

def get_rff_pipelined_logistic_classification(out_d,sigma=1,logistic_args={},matrix_seed = None,random_state=None):    
    """returns a pipeline that combines standard scaling, projection to a random fourier feature, and logistic classification into a single pipelined estimator that can then be bagged.  

    """
    full_estimator = Pipeline([('standardscaler',StandardScaler()),('rff_regression',RFFLogisticRegression(out_d,matrix_seed,random_state,logistic_params = logistic_args))])
    return full_estimator


def get_rff_pipelined_linear_regression(out_d,sigma=1,linear_args ={},matrix_seed = None,random_state= None)
    full_estimator = Pipeline([('standardscaler',StandardScaler()),('rff_regression',RFFLinearRegression(out_d,matrix_seed,random_state,linear_params = linear_args))])
    return full_estimator



### if we want random weights every time, we need a estimator object with random state. 

class RFFLogisticRegression(BaseEstimator,ClassifierMixin):
    """Builds out a random features projection to apply to input data before standard logistic regression. 

    Parameters
    ----------
    n_features : int
        number of features to project to (default: 1). 
    sigma : float    
        variance of the normal distribution used to sample projection weights (default: 1) 
    random_state : int or None     
        random state used to control randomness of the weight initialization. If provided in a Bagging classifier, resulting bagged estimators will have different projection matrices.
    matrix_seed : int or None    
        random state used to directly fix a projection matrix. If provided in a Bagging classifer, results in bagged estimators with the same projection matrix.  
    logistic_params: dict    
        dictionary of parameters to pass to LogisticRegression.
    """
    def __init__(self,n_features=1,sigma=1,random_state = None,matrix_seed = None,logistic_params = {}):
        self.n_features = n_features
        self.random_state = random_state
        self.sigma = 1
        self.logistic_params = logistic_params
        self.matrix_seed = matrix_seed
    
    def project(self,data):
        """Assumes data is of shape (batch,features)

        """
        return np.cos(np.matmul(data,self.matrix.T)+self.offset)

    def decision_function(self,X):
        return self.lr.decision_function(self.project(X))

    def densify(self):
        return self.lr.densify()

    def fit(self,X,y,sample_weight = None):
        self.lr = LogisticRegression(**self.logistic_params)
        input_dim = X.shape[1]
        if self.matrix_seed is None:
            self.matrix =  sklearn.utils.check_random_state(self.random_state).randn(self.n_features,input_dim)/np.sqrt(self.sigma*input_dim)
            self.offset = sklearn.utils.check_random_state(self.random_state).uniform(0,2*np.pi,size = self.n_features)
        else:
            self.matrix =  sklearn.utils.check_random_state(self.matrix_seed).randn(self.n_features,input_dim)/np.sqrt(self.sigma*input_dim)
            self.offset = sklearn.utils.check_random_state(self.matrix_seed).uniform(0,2*np.pi,size = out_d)
        self.lr.fit(self.project(X),y,sample_weight)
        self.classes_ = self.lr.classes_
        return self

    def predict(self,X):
        return self.lr.predict(self.project(X))

    def predict_log_proba(self,X):
        return self.lr.predict_log_proba(self.project(X))

    def predict_proba(self,X):        
        return self.lr.predict_proba(self.project(X))

    def score(self,X,y,sample_weight = None):
        return self.lr.score(self.project(X),y,sample_weight)

class RFFLinearRegression(BaseEstimator,ClassifierMixin):
    """Builds out a random features projection to apply to input data before standard linear regression. 

    Parameters
    ----------
    n_features : int
        number of features to project to (default: 1). 
    sigma : float    
        variance of the normal distribution used to sample projection weights (default: 1) 
    random_state : int or None     
        random state used to control randomness of the weight initialization. If provided in a Bagging classifier, resulting bagged estimators will have different projection matrices.
    matrix_seed : int or None    
        random state used to directly fix a projection matrix. If provided in a Bagging classifer, results in bagged estimators with the same projection matrix.  
    logistic_params: dict    
        dictionary of parameters to pass to LinearRegression.
    """
    def __init__(self,n_features=1,sigma=1,random_state = None,matrix_seed = None,logistic_params = {}):
        self.n_features = n_features
        self.random_state = random_state
        self.sigma = 1
        self.linear_params = linear_params
        self.matrix_seed = matrix_seed
    
    def project(self,data):
        """Assumes data is of shape (batch,features)

        """
        return np.cos(np.matmul(data,self.matrix.T)+self.offset)

    #def decision_function(self,X):
    #    return self.lr.decision_function(self.project(X))

    #def densify(self):
    #    return self.lr.densify()

    def fit(self,X,y,sample_weight = None):
        self.lr = LinearRegression(**self.linear_params)
        input_dim = X.shape[1]
        if self.matrix_seed is None:
            self.matrix =  sklearn.utils.check_random_state(self.random_state).randn(self.n_features,input_dim)/np.sqrt(self.sigma*input_dim)
            self.offset = sklearn.utils.check_random_state(self.random_state).uniform(0,2*np.pi,size = self.n_features)
        else:
            self.matrix =  sklearn.utils.check_random_state(self.matrix_seed).randn(self.n_features,input_dim)/np.sqrt(self.sigma*input_dim)
            self.offset = sklearn.utils.check_random_state(self.matrix_seed).uniform(0,2*np.pi,size = out_d)
        self.lr.fit(self.project(X),y,sample_weight)
        self.classes_ = self.lr.classes_
        return self

    def predict(self,X):
        return self.lr.predict(self.project(X))

    #def predict_log_proba(self,X):
    #    return self.lr.predict_log_proba(self.project(X))

    #def predict_proba(self,X):        
    #    return self.lr.predict_proba(self.project(X))

    def score(self,X,y,sample_weight = None):
        return self.lr.score(self.project(X),y,sample_weight)



            
