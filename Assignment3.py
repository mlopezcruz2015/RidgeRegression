import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataframe = pd.read_csv('/Users/mlopezcruz2015/Documents/AIProgrammingAssignments/TrainingData_N183_p10.csv', ',',
                        usecols=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10', 'Ancestry'])

# Global Variables
X_headers = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10']
Y_header = 'Ancestry'

learningRateA = 10**-5
tuning_parameters_lambda = []
exponents = np.linspace(-4, 4, 9)
for n in exponents:
    tuning_parameters_lambda.append(10 ** n)


x = dataframe[X_headers]
y = dataframe[Y_header]

x = x/x.std()

x.insert(loc=0, column='K', value=1)

y = y.map({'African':1, 'European':2, 'EastAsian':3, 'Oceanian':4, 'NativeAmerican':5})
y_mapped = y.map({1:np.array([1,0,0,0,0]), 2:np.array([0,1,0,0,0]), 3:np.array([0,0,1,0,0]), 4:np.array([0,0,0,1,0]), 5:np.array([0,0,0,0,1])})
print(y_mapped)

b = np.zeros((183, 5))
y = np.zeros((183, 5))

print('done')
