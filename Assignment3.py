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
y_temp = dataframe[Y_header]

x = x.to_numpy()
x = x-x.mean()
x = x/x.std()

x = np.append(np.ones((183, 1)), x, 1)

y_temp = y_temp.map({'African':1, 'European':2, 'EastAsian':3, 'Oceanian':4, 'NativeAmerican':5})
y_mapped = y_temp.map({1:np.array([1,0,0,0,0]), 2:np.array([0,1,0,0,0]), 3:np.array([0,0,1,0,0]), 4:np.array([0,0,0,1,0]), 5:np.array([0,0,0,0,1])})
y_mapped = y_mapped.to_numpy()
print(y_mapped)

y = np.zeros((183, 5))

for i in range(183):
    for j in range(5):
        y[i][j] = y_mapped[i][j]


b = np.zeros((11, 5))
b = b.astype(float)

b_array = []

for l in range(len(tuning_parameters_lambda)):
    for v in range(10000):

        u = np.exp(np.matmul(x, b))

        p = np.zeros((u.shape[0], u.shape[1]))
        for i in range(183):
            for j in range(5):
                p[i][j] = u[i][j]/(np.sum(u[[i]]))

        z = np.zeros((11, 5))
        for a in range(5):
            z[0][a] = b[0][a]

        previous_b = b
        b = b + learningRateA*(np.subtract(np.dot(np.transpose(x), (np.subtract(y,p))), np.dot(2 * tuning_parameters_lambda[l], np.subtract(b, z))))

        if np.array_equal(previous_b, b):
            break
    b_array.append(b)

plt.xlabel("Lambda (10^x)")
plt.ylabel("Standardized Coefficients (Beta)")
plt.plot(pd.DataFrame(b_array[0], index=exponents))
plt.show()

print('done')
