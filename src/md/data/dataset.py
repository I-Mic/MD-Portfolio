
from typing import Sequence
import numpy as np


class Dataset:
    """
    Dataset class for storing and manipulating data.

    Parameters:
        X (np.ndarray, optional): The input features. Defaults to None.
        y (np.ndarray, optional): The target labels. Defaults to None.
        features (Sequence[str], optional): The feature names. Defaults to None.
        label (str, optional): The label name. Defaults to None.

    Attributes:
        X (np.ndarray): The input features.
        y (np.ndarray): The target labels.
        features (Sequence[str]): The feature names.
        label (str): The label name.
        all_cols (list): The names of all columns (features and label).
        data (np.ndarray): The dataset containing both numerical and categorical data.
        numerical_cols (list): The indices of numerical columns in the dataset.
        categorical_cols (list): The indices of categorical columns in the dataset.
        encode_categories (dict): A dictionary mapping categorical column indices to their categories.

    """

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



    def __get_categories(self, data, columns):
        """
        Retrieves the categories for categorical columns.

        Parameters:
            data (np.ndarray): The dataset.
            columns (List[int]): The indices of the categorical columns.

        Returns:
            dict: A dictionary where the keys are the column indices and the values are the unique categories.
        """

        categories = {}
        for c in range(len(columns)):
            column = data.T[c]
            uniques = np.unique(column)
            categories[c] = np.delete(uniques, uniques == '')
        return categories
        
        
    def __label_encode(self, data, categorical_columns):
        """
        Performs label encoding for categorical columns.

        Parameters:
            data (np.ndarray): The dataset.
            categorical_columns (List[int]): The indices of the categorical columns.

        Returns:
            np.ndarray: The encoded dataset.
            dict: A dictionary where the keys are the column indices and the values are the unique categories.

        """

        categories = self.__get_categories(data, categorical_columns)
        encoded_data = np.full(data.shape, np.nan)
    
        for k in categories:
            cats = categories[k]
            for c in range(len(cats)):
                cat = cats[c]
                dt = np.transpose((data.T[k] == cat)).nonzero()
                encoded_data.T[k, dt] = c

        return encoded_data, categories

    
    def __get_feature_type(self, value):
        """
        Determines the type of a feature based on its value.

        Parameters:
            value: The value of the feature.

        Returns:
            str: The type of the feature ('numerical', 'categorical') or None if the type cannot be determined.

        """

        try:
            arr = np.array([float(value)])
        except ValueError:
            arr = np.array([str(value)])

        if np.issubdtype(arr.dtype, np.floating): return 'numerical'
        if np.issubdtype(arr.dtype, np.dtype('U')): return 'categorical'

        return None
    
    def __get_types(self, filename, sep):
        """
        Retrieves the feature types from a CSV file.

        Parameters:
            filename (str): The path to the CSV file.
            sep (str): The delimiter used in the CSV file.

        Returns:
            List[str]: The names of the features.
            List[int]: The indices of the numerical columns.
            List[int]: The indices of the categorical columns.

        """

        with open(filename) as file:
            features = file.readline().rstrip().split(sep)

            row = file.readline().rstrip().split(sep)
            numericals = []
            categoricals = []

            for i in range(len(row)):
                column = row[i]
                dtype = self.__get_feature_type(column)

                if dtype == 'numerical':
                    numericals.append(i)
                elif dtype == 'categorical':
                    categoricals.append(i)
            
            return features, numericals, categoricals

    def read_csv(self, filename,label=None, sep = ","):
        """
        Reads a dataset from a CSV file.

        Parameters:
            filename (str): The path to the CSV file.
            label (str, optional): The name of the label. Defaults to None.
            sep (str, optional): The delimiter used in the CSV file. Defaults to ",".

        """

        #Gets type of each feature
        features, numericals, categoricals = self.__get_types(filename, sep)
        #Saves numerical data and categorical data from dataset
        numerical_data = np.genfromtxt(filename, delimiter=sep, usecols=numericals)
        categorical_data = np.genfromtxt(filename, delimiter=sep, dtype='U', usecols=categoricals)

        if len(categorical_data.shape) == 1:
            categorical_data = np.reshape(categorical_data, (categorical_data.shape[0], 1))
        #Encodes all categorical data
        encoded_data, categories = self.__label_encode(categorical_data, categoricals)
        if len(categoricals) > 0:
            # Concatenate numerical and encoded categorical data
            data = np.concatenate((numerical_data.T, encoded_data.T)).T
            # Fill the dataset array with NaN values
            data = np.full((numerical_data.shape[0], numerical_data.shape[1] + encoded_data.shape[1]), np.nan)
            # Assign numerical data and encoded categorical data to their respective columns
            data.T[numericals] = numerical_data.T
            data.T[categoricals] = encoded_data.T

        else:
            # Only numerical data, no categorical data
            data = np.concatenate((numerical_data.T)).T
            data = np.full((numerical_data.shape[0], numerical_data.shape[1]), np.nan)
            data.T[numericals] = numerical_data.T 

        self.all_cols = features.copy()
        self.data = data[1:].copy()
        
        #Uses last dataset feature as label by default
        if label == None:
            self.features = features[:-1]
            self.label = features[-1]
            self.X = data[1:,0:-1]
            self.y = data[1:,-1]
        #In case another label is determined
        else:
            self.label = label
            self.y = data[1:,features.index(self.label)].T
            self.X = np.delete(data, features.index(self.label), axis=1)
            self.X = self.X[1:]
            self.features = features.copy()
            self.features.remove(self.label)

        
        self.numerical_cols = numericals
        self.categorical_cols = categoricals
        self.encode_categories = categories
        
    def read_tsv(self,filename,label=None):
        """
        Reads a dataset from a TSV file.

        Parameters:
            filename (str): The path to the TSV file.
            label (str, optional): The name of the label. Defaults to None.

        """

        self.read_csv(self,filename,label,sep='\t')



    def write_csv(self,file,delimiter=','):
        """
        Writes the dataset to a CSV file.

        Parameters:
            file (file-like object): The file object to write to.
            delimiter (str, optional): The delimiter to use in the CSV file. Defaults to ','.

        """

        data = np.hstack((self.X, self.y.reshape(-1, 1)))
        header = self.features + [self.label]
        np.savetxt(file, data, delimiter=delimiter, header=delimiter.join(header),fmt='%s',comments='')

    def write_tsv(self, file):
        """
        Writes the dataset to a TSV file.

        Parameters:
            file (file-like object): The file object to write to.

        """

        self.write_csv(file,'\t')

    def get_X(self):
        """
        Returns the feature matrix X.

        Returns:
            numpy.ndarray: The feature matrix X.

        """
        return self.X

    def set_X(self,new):
        """
        Sets the feature matrix X.

        Parameters:
            new (numpy.ndarray): The new feature matrix X.

        """
        self.X = new

    def get_y(self):
        """
        Returns the label vector y.

        Returns:
            numpy.ndarray: The label vector y.

        """
        return self.y
    
    def get_Xy(self):
        """
        Returns the feature matrix X and label vector y.

        Returns:
            tuple: A tuple containing the feature matrix X and label vector y.

        """
        return self.X,self.y
    
    def get_features(self):
        """
        Returns the list of feature names.

        Returns:
            list: The list of feature names.

        """
        return self.features

    def get_label(self):
        """
        Returns the name of the label.

        Returns:
            str: The name of the label.

        """
        return self.label
    
    def get_classes(self):
        """
        Returns the unique classes in the label vector.

        Returns:
            numpy.ndarray: The unique classes in the label vector.

        Raises:
            ValueError: If the dataset does not have a label.

        """

        if self.y is None:
            raise ValueError("Dataset does not have a label")
        return np.unique(self.y)

    def describe(self):
        """
        Provides simple analytical data for each type of entry in the dataset.

        """
        #For categorical data returns nr of unique values and most frequent value
        for i in range(len(self.categorical_cols)):
            print(self.all_cols[self.categorical_cols[i]])
            var = self.data[:, self.categorical_cols[i]]   
            unique_vals, counts = np.unique(var, return_counts=True)
            print(" -Quantidade de valores únicos: ",len(unique_vals))
            print(" -Valor mais frequente: ",np.argmax(counts))
        #For numerical data returns mean median standard deviation min and max
        for i in range(len(self.numerical_cols)):
            print(self.all_cols[self.numerical_cols[i]])
            var = self.data[:, self.numerical_cols[i]] 
            print(" -Média: ",np.nanmean(var))
            print(" -Mediana: ",np.nanmedian(var))
            print(" -Desvio padrão: ",np.nanstd(var))
            print(" -Minimo: ",np.nanmin(var))
            print(" -Máximo: ",np.nanmax(var))


   

    def count_nulls(self):
        """
        Counts the number of null values in the dataset.

        """

        null_count = np.zeros(self.X.shape[1], dtype=int)
        #Nulls count for each feature
        for i in range(self.X.shape[1]):
            for val in self.X[:, i]:
                if isinstance(val, str):
                    if val.strip() == '' or val == None:
                        null_count[i] += 1
                elif np.isnan(val):
                    null_count[i] += 1


        #Nulls count for label
        for i in range(len(self.features)):
            print(self.features[i],  "- valores nulos:",  null_count[i])

    def count_unique(self):
        """
        Counts the number of unique values in each feature of the dataset.

        Returns:
            numpy.ndarray: The number of unique values for each feature.

        """
        # Counts the number of unique values in each feature
        unique_count = np.zeros(self.X.shape[1], dtype=int)
        for i in range(self.X.shape[1]):
            unique_count[i] = len(np.unique(self.X[:, i]))
        return unique_count
    

    def replace_to_null(self,value):
        """
        #Replaces a given value to a np.nan

        """
        self.X = np.where(self.X==value,np.nan,self.X)
        self.y = np.where(self.y==value,np.nan,self.y)

    def replace_nulls(self, method, index=-1):
        """
        Replaces nulls with a given method

        """
        #By default replaces nulls for all features
        if index < 0:
            #Replaces nulls with the mode of the feature
            if method == "mode":
                for i in range(self.X.shape[1]):
                    var = self.X[:,i]                  
                    _, counts = np.unique(var, return_counts=True)
                    mode = np.argmax(counts)
                    self.X[:,i] = np.where(np.isnan(var),mode,var)
            #Replaces nulls with the mean of the feature
            elif method == "mean":
                for i in range(self.X.shape[1]):
                    var = self.X[:,i]
                    mean = np.nanmean(var)
                    self.X[:,i] = np.where(np.isnan(var),mean,var)

            else: print("Method not recognized")

        #Can also replace null for a single determined feature
        elif self.X.shape[1] > index:                
            var = self.X[:,index]
            #Replaces nulls with mode
            if method == "mode":                  
                _, counts = np.unique(var, return_counts=True)
                mode = np.argmax(counts)
                self.X[:,index] = np.where(np.isnan(var),mode,var)
            #Replaces nulls with mean
            elif method == "mean":
                mean = np.nanmean(var)
                self.X[:,index] = np.where(np.isnan(var),mean,var)

            else: print("Method not recognized")

        else: print("Index out of bounds")

        
