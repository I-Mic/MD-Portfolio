
from typing import Sequence
import numpy as np


class Dataset:
    def __init__(self, X: np.ndarray = None, y: np.ndarray = None, features: Sequence[str] = None, label: str = None):
        
        if features is None and X is not None:
            features = [str(i) for i in range(X.shape[1])]
        elif features is not None:
            features = list(features)

        if y is not None and label is None:
            label = "y"

        self.X = X
        self.y = y
        self.features = features
        self.label = label

    def __get_feature_type(self, value):
        try:
            arr = np.array([float(value)])
        except ValueError:
            arr = np.array([str(value)])

        if np.issubdtype(arr.dtype, np.floating): return 'numerical'
        if np.issubdtype(arr.dtype, np.dtype('U')): return 'categorical'

        return None
    
    def __get_types(self, filename, sep):
        with open(filename) as file:
            features = file.readline().rstrip().split(sep)

            row = file.readline().rstrip().split(sep)
            numericals = []
            categoricals = []

            for i in range(len(row)):
                col = row[i]
                dtype = self.__get_feature_type(col)

                if dtype == 'numerical':
                    numericals.append(i)
                elif dtype == 'categorical':
                    categoricals.append(i)
            
            return features, numericals, categoricals
        
    def __get_categories(self, data, cols):
        categories = {}
        for c in range(len(cols)):
            col = data.T[c]
            uniques = np.unique(col)
            categories[c] = np.delete(uniques, uniques == '')
        return categories
        
    def __label_encode(self, data, categorical_columns):
        categories = self.__get_categories(data, categorical_columns)
        encoded_data = np.full(data.shape, np.nan)
    
        for k in categories:
            cats = categories[k]
            for c in range(len(cats)):
                cat = cats[c]
                dt = np.transpose((data.T[k] == cat)).nonzero()
                encoded_data.T[k, dt] = c

        return encoded_data, categories

    def read_csv(self, filename,label=None, sep = ","):
        features, numericals, categoricals = self.__get_types(filename, sep)
            
        numerical_data = np.genfromtxt(filename, delimiter=sep, usecols=numericals)
        categorical_data = np.genfromtxt(filename, delimiter=sep, dtype='U', usecols=categoricals)
        
        if len(categorical_data.shape) == 1:
            categorical_data = np.reshape(categorical_data, (categorical_data.shape[0], 1))

        encoded_data, categories = self.__label_encode(categorical_data, categoricals)
        data = np.concatenate((numerical_data.T, encoded_data.T)).T
        data = np.full((numerical_data.shape[0], numerical_data.shape[1] + encoded_data.shape[1]), np.nan)
        data.T[numericals] = numerical_data.T
        data.T[categoricals] = encoded_data.T

        self.all_cols = features.copy()
        self.data = data[1:].copy()

        if label == None:
            self.features = features[:-1]
            self.label = features[-1]
            self.X = data[1:,0:-1]
            self.y = data[1:,-1]

        else:
            self.label = label
            self.y = data[1:,features.index(self.label)].T
            self.X = np.delete(data, features.index(self.label), axis=1)
            self.X = self.X[1:]
            self.features = features
            self.features.remove(self.label)

        
        self.numerical_cols = numericals
        self.categorical_cols = categoricals
        self.encode_categories = categories
        


    def read_tsv(self,filename,label=None):
        self.read_csv(self,filename,label,sep='t')

    def write_csv(self,file,delimiter=','):
        data = np.hstack((self.X, self.y.reshape(-1, 1)))
        header = self.features + [self.label]
        np.savetxt(file, data, delimiter=delimiter, header=delimiter.join(header),fmt='%s',comments='')

    def write_tsv(self, file):
        self.write_csv(file,'\t')

    def get_X(self):
        return self.X

    def set_X(self,new):
        self.X = new

    def get_y(self):
        return self.y
    
    def get_features(self):
        return self.features

    def get_label(self):
        return self.label

    def describe(self):
        #Descreve as variaveis de entrada e saida
        for i in range(len(self.categorical_cols)):
            print(self.all_cols[self.categorical_cols[i]])
            var = self.data[:, self.categorical_cols[i]]   
            unique_vals, counts = np.unique(var, return_counts=True)
            print(" -Quantidade de valores únicos: ",len(unique_vals))
            print(" -Valor mais frequente: ",np.argmax(counts))
            
        for i in range(len(self.numerical_cols)):
            print(self.all_cols[self.numerical_cols[i]])
            var = self.data[:, self.numerical_cols[i]] 
            print(" -Média: ",np.nanmean(var))
            print(" -Mediana: ",np.nanmedian(var))
            print(" -Desvio padrão: ",np.nanstd(var))
            print(" -Minimo: ",np.nanmin(var))
            print(" -Máximo: ",np.nanmax(var))


   

    def count_nulls(self):
        #Conta os valores nulos nas variaveis de entrada e saida
        null_count = np.zeros(self.X.shape[1], dtype=int)
        
        for i in range(self.X.shape[1]):
            for val in self.X[:, i]:
                if isinstance(val, str):
                    if val.strip() == '' or val == None:
                        null_count[i] += 1
                elif np.isnan(val):
                    null_count[i] += 1

        for val in self.y:
            if isinstance(val, str):
                if val == None:
                    null_count[-1] += 1
            elif np.isnan(val):
                null_count[-1] += 1
        
        for i in range(len(self.features)):
            print(self.features[i],  "- valores nulos:",  null_count[i])
        print(self.label, "- valores nulos:", null_count[-1])

    def replace_to_null(self,value):
        #Substitui determinados valores para np.nan
        self.X = np.where(self.X==value,np.nan,self.X)
        self.y = np.where(self.y==value,np.nan,self.y)

    def replace_nulls(self, method, index=-1):
        if index < 0:
            if method == "mode":
                for i in range(self.X.shape[1]):
                    var = self.X[:,i]                  
                    _, counts = np.unique(var, return_counts=True)
                    mode = np.argmax(counts)
                    self.X[:,i] = np.where(np.isnan(var),mode,var)
            elif method == "mean":
                for i in range(self.X.shape[1]):
                    var = self.X[:,i]
                    mean = np.nanmean(var)
                    self.X[:,i] = np.where(np.isnan(var),mean,var)

            else: print("Method not recognized")

        elif self.X.shape[1] > index:                
            var = self.X[:,index]
            if method == "mode":                  
                _, counts = np.unique(var, return_counts=True)
                mode = np.argmax(counts)
                self.X[:,index] = np.where(np.isnan(var),mode,var)
            elif method == "mean":
                mean = np.nanmean(var)
                self.X[:,index] = np.where(np.isnan(var),mean,var)

            else: print("Method not recognized")

        else: print("Index out of bounds")

        
