
# Copyright (c) 2019, Ahmed M. Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from __future__ import absolute_import, division, print_function

import numpy as np

from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor

import warnings
warnings.filterwarnings("ignore")


class baseImputer:
    """
    Base class for constructing an imputation method on the pipeline. Default imputer is the mean imputation strategy.
    Includes R wrappers for various imputation methods.
    
    Available imputers:
    ------------------
    sk-learn: mean, median and most frequent imputation strategies.
    
    Attributes:
    ----------
    _hyperparameters: a dictionary of hyperparameters for the class instance along with their current values.
    _mode: the selected imputation algorithm for this class instance.
    model: imputation model object.
    
    Methods:
    -------
    fit: a method for imputing missing data using the selected algorithm and hyperparameters
    
    """

    def __init__(self,**kwargs):
        """
        Class constructor. 
        Initialize an Imputer object. 
    
        :_mode: imputation algorithm (options: 'mean', 'median', 'most_frequent')
        :_hyperparameters: hyper-parameter setting for the imputer 
        
        """ 
        self._model_list       = ['mean', 'median', 'most_frequent', 'iterative_k_neighbors', 'iterative_extra_trees', 
                                    'iterative_decision_tree', 'iterative_bayesian_ridge']
        self._hyperparameters  = {} 
        self._mode             = 'mean' 
        self.MI                = False   
        self.est               = None
        self.kwargs            = kwargs

        self.is_init_r_system = False
        
        # Set defaults and catch exceptions
        self.__acceptable_keys_list = ['_mode', '_hyperparameters', 'est']
        
        try:
            if(len(kwargs) > 0):
                
                [self.__setattr__(key, kwargs.get(key)) for key in self.__acceptable_keys_list]
            
            #if self._mode not in self._model_list:
                
                #raise ValueError("Unrecognized imputation model! Default mean imputation will be used.")
        
        except ValueError as valerr:
            
            print(valerr)
            self._hyperparameters  = {} 
            self._mode             = 'mean'
        
        # Set imputation model object
        
        self.set_model()
        self.set_hyperparameters()
        
        
    def set_model(self):
        """
        Creates an imputation model object and assigns it to the "model" attribute.
        """
        if (self._mode in ['mean', 'median', 'most_frequent']):
            self.model   = SimpleImputer(strategy=self._mode)
        else:
            self.model = IterativeImputer(estimator= self.est)
                   
    
    def set_hyperparameters(self):
        """
        Set the imputation model hyper-parameters.
        """  
        hyper_dict   = {'mean': [], 'median': [], 'most_frequent': [], 'iterative_k_neighbors':[],
                     'iterative_extra_trees': [], 'iterative_decision_tree': [], 'iterative_bayesian_ridge':[]}
        
        _hyp__input  = (self.kwargs.__contains__('_hyperparameters')) # Hyperparameters input FLAG
        
        missing_hyp     = []
        missing_flg     = False
        self.hyper_dict = hyper_dict 
        
        # cleanup inputs by deleting wrongly provided hyperparameters
        
        if _hyp__input:
            
            hyper_keys = list(self._hyperparameters.keys())
            
            # clean wrong inputs
            
            for u in range(len(hyper_keys)):
                
                if hyper_keys[u] not in hyper_dict[self._mode]:
                    self._hyperparameters.pop(hyper_keys[u], None)
            
            for u in range(len(hyper_dict[self._mode])):
                
                if hyper_dict[self._mode][u] not in hyper_keys:
                    missing_hyp.append(hyper_dict[self._mode][u])
                    missing_flg = True
        else:
            missing_hyp = hyper_dict[self._mode]
            missing_flg = True
            
        # set defaults if no input provided
        
        if self._mode in ['mean','median','most_frequent', 'iterative_k_neighbors', 'iterative_extra_trees', 
                        'iterative_decision_tree', 'iterative_bayesian_ridge']:
            self._hyperparameters  = {}
            
        elif missing_flg:
            
            # create an empty hyperparameters dictionary 
            self._hyperparameters = dict.fromkeys(hyper_dict[self._mode])
            
            for v in range(len(missing_hyp)):
                self._hyperparameters[missing_hyp[v]] = default_dict[self._mode][missing_hyp[v]]
                    
    def fit(self, X):
        """
        Impute missing data using the current instance of the imputation model. 
        :X: Input features with missing data.
        """  
        
        X = np.array(X)
        X = self.model.fit_transform(X)
        return X

    def init_r_sytem(self):

        if not self.is_init_r_system:
            self.collect_Rpackage_imputation_()
            self.is_init_r_system = True


class mean:
    """ Mean imputation algorithm."""
    
    
    def __init__(self): 
        
        self.model_type    = 'imputer'
        self.name          = 'mean'
        self.MI            = False
        self.model         = baseImputer(_mode=self.name)
        
    def fit_transform(self, X):
        
        return self.model.fit(X)
        
    def get_hyperparameter_space(self):
            
        hyp_               = []
        
        return hyp_
    

class median:
    """ Median imputation algorithm."""
    
    
    def __init__(self): 
        
        self.model_type    = 'imputer'
        self.name          = 'median'
        self.MI            = False
        self.model         = baseImputer(_mode=self.name)
        
    def fit_transform(self, X):
        
        return self.model.fit(X)
        
    def get_hyperparameter_space(self):
            
        hyp_               = []
        
        return hyp_     
    
    
class most_frequent:
    """ Most frequent imputation algorithm."""
    
    
    def __init__(self): 
        
        self.model_type    = 'imputer'
        self.name          = 'most_frequent'
        self.MI            = False
        self.model         = baseImputer(_mode=self.name)
        
    def fit_transform(self, X):
        
        return self.model.fit(X)
        
    def get_hyperparameter_space(self):
            
        hyp_               = []
        
        return hyp_         

class iterative_bayesian_ridge:
    """ iterative imputation algorithm using bayesian ridge estimator."""
    
    def __init__(self): 
        
        self.model_type    = 'imputer'
        self.name          = 'iterative_bayesian_ridge'
        self.MI            = False
        self.estimator     = BayesianRidge()
        self.model         = baseImputer(_mode=self.name, est=self.estimator)
        
    def fit_transform(self, X):
        
        return self.model.fit(X)
        
    def get_hyperparameter_space(self):
            
        hyp_               = []
        
        return hyp_         

class iterative_decision_tree:
    """ iterative imputation algorithm using decision tree estimator."""
    
    def __init__(self): 
        
        self.model_type    = 'imputer'
        self.name          = 'iterative_decision_tree'
        self.MI            = False
        self.estimator     = DecisionTreeRegressor(max_features='sqrt', random_state=0)
        self.model         = baseImputer(_mode=self.name, est=self.estimator)
        
    def fit_transform(self, X):
        
        return self.model.fit(X)
        
    def get_hyperparameter_space(self):
            
        hyp_               = []
        
        return hyp_         

class iterative_extra_trees:
    """ iterative imputation algorithm using extra trees estimator."""
    
    def __init__(self): 
        
        self.model_type    = 'imputer'
        self.name          = 'iterative_extra_trees'
        self.MI            = False
        self.estimator     = ExtraTreesRegressor(n_estimators=10, random_state=0)
        self.model         = baseImputer(_mode=self.name, est=self.estimator)
        
    def fit_transform(self, X):
        
        return self.model.fit(X)
        
    def get_hyperparameter_space(self):
            
        hyp_               = []
        
        return hyp_         

class iterative_k_neighbors:
    """ iterative imputation algorithm using k-neighbors estimator."""
    
    def __init__(self): 
        
        self.model_type    = 'imputer'
        self.name          = 'iterative_k_neighbors'
        self.MI            = False
        self.estimator     = KNeighborsRegressor(n_neighbors=15)
        self.model         = baseImputer(_mode=self.name, est=self.estimator)
        
    def fit_transform(self, X):
        
        return self.model.fit(X)
        
    def get_hyperparameter_space(self):
            
        hyp_               = []
        
        return hyp_         