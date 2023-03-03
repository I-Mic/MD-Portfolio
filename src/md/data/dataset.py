import numpy as np
import pandas as pd

class Dataset:
    
    def __init__(self):
        self.X = np.array([])
        self.y = np.array([])
        self.features = []
        self.label = ''

    def read_csv(self,file,label=None,delimiter=','):
        if label == None:
            data = np.genfromtxt(file,delimiter=delimiter,names=True,dtype=None,encoding=None)
            self.features = list(data.dtype.names)
            self.label = self.features[len(self.features)-1]
            self.features.remove(self.label)
            
        else:
            data = np.genfromtxt(file,delimiter=delimiter,names=True,dtype=None,encoding=None)
            self.label = label
            self.features = list(data.dtype.names)
            self.features.remove(label)
        
        self.X = np.vstack([data[f] for f in self.features], dtype=object).T
        self.y = data[self.label]
    
        
    def read_tsv(self,file,label=None):
        self.read_csv(file,label,'\t')


    def write_csv(self,file,delimiter=','):
        data = np.hstack((self.X, self.y.reshape(-1, 1)))
        header = self.features + [self.label]
        fmt = [ "%.18e" if col.dtype.kind in {'f', 'c'} else "%s" for col in data.T ]
        np.savetxt(file, data, delimiter=delimiter, header=delimiter.join(header),fmt=fmt, comments='')

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
        for i, name in enumerate(self.features):
            print(name)
            var = self.X[:, i]
            for v in var:
                if v != np.nan:
                    first_non_null = v
                    break
                
            if isinstance(first_non_null,str):
                unique_vals, counts = np.unique(var[var == var], return_counts=True)
                print(" -Quantidade de valores únicos: ",len(unique_vals))
                print(" -Valor mais frequente: ",unique_vals[np.argmax(counts)])
            else:
                print(" -Média: ",np.nanmean(var))
                print(" -Mediana: ",np.nanmedian(var))
                print(" -Desvio padrão: ",np.nanstd(var))
                print(" -Minimo: ",np.nanmin(var))
                print(" -Máximo: ",np.nanmax(var))

        if self.label != None:
            print(self.label)
            var = self.y
            for v in var:
                if v != np.nan:
                    first_non_null = v
            if isinstance(first_non_null,str):
                unique_vals, counts = np.unique(var, return_counts=True)
                print(" -Quantidade de valores únicos: ",len(unique_vals))
                print(" -Valor mais frequente: ",unique_vals[np.argmax(counts)])
            else:
                print(" -Média: ",np.mean(var))
                print(" -Mediana: ",np.median(var))
                print(" -Desvio padrão: ",np.std(var))
                print(" -Minimo: ",np.min(var))
                print(" -Máximo: ",np.max(var))
        

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

    def replace_nulls(self, value):
        #Substiui todos os valores nulos por um determinado valor
        self.X = np.where(self.X != self.X, value, self.X)
        self.y = np.where(self.y != self.y, value, self.y)
    
    def replace_nulls_automatic(self):
        #Automaticamente substitui valores nulos pela 
        # média para valores numéricos e 
        # valor mais frequente para valores categóricos 
        for i,n in enumerate(self.features):
            var = self.X[:, i]
            for v in var:
                if v != np.nan:
                    first_non_null = v
            if isinstance(first_non_null,str):
                unique_vals, counts = np.unique(var[var==var], return_counts=True)
                self.X[:,i] = np.where(var != var,unique_vals[np.argmax(counts)],var)
            else:
                val = np.nanmean(var)
                self.X[:,i] = np.where(var != var,val,var)

            var = self.y
            for v in var:
                if v != np.nan:
                    first_non_null = v
            if isinstance(first_non_null,str):
                unique_vals, counts = np.unique(var[var==var], return_counts=True)
                self.y = np.where(var != var,unique_vals[np.argmax(counts)],var)
            else:
                val = np.nanmean(var)
                self.y = np.where(var != var,val,var)
    
    def replace_to_null(self,value):
        #Substitui determinados valores para np.nan
        self.X = np.where(self.X==value,np.nan,self.X)
        self.y = np.where(self.y==value,np.nan,self.y)
        