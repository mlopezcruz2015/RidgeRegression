import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataframe = pd.read_csv('/Users/mlopezcruz2015/PycharmProjects/AIProgrammingAssignment1/Credit_N400_p9.csv', ',',
                        usecols=['Income', 'Limit', 'Rating', 'Cards', 'Age', 'Education', 'Gender', 'Student',
                                 'Married', 'Balance'])

# Replace categorical data
dataframe = dataframe.replace({'Gender': {'Male': 1, 'Female': 0},
                               'Student': {'Yes': 1, 'No': 0},
                               'Married': {'Yes': 1, 'No': 0}})

def center_data(df):
    temp = df - df.mean()
    return temp


def standardize_data(df):
    temp = center_data(df)
    temp = temp / temp.std()
    return temp

# Algorithm
def algorithm(X, y, learning_rate, tuning_parameter):

    # Randomly initialize the parameter vector
    betas = np.random.uniform(-1, 1, size=9)

    for i in range(1000):
        for k in range(len(betas)):
            Ak  = np.dot(X[:, k].T, np.add(y - np.dot(X, betas), np.multiply(betas[k], X[:, k])))
            betas[k] = (np.sign(Ak) * max(0, np.abs(Ak) - (learning_rate*(1-tuning_parameter)/2)) / np.add(b[k], learning_rate*tuning_parameter))

    return betas

# Global Variables
X_headers = ['Income', 'Limit', 'Rating', 'Cards', 'Age', 'Education', 'Gender', 'Student', 'Married']
Y_header = 'Balance'
learningRateA = 10 ** -5

tuning_parameters_lambda = []
exponents = np.linspace(-2, 6, 9)
for n in exponents:
    tuning_parameters_lambda.append(10 ** n)

tuning_parameters_alpha = [0, .2, .4, .6, .8, 1]

# Centering and Scaling X (Standardizing)
X_numpy = dataframe[X_headers]
X_numpy = standardize_data(X_numpy).to_numpy()

# Centering Y
Y_numpy = dataframe[Y_header]
Y_numpy = center_data(Y_numpy).to_numpy()

# Precompute Bk
b = np.zeros(9)
X2 = np.square(X_numpy)
b = X2.sum(axis=0)

np.seterr(invalid='ignore')
B_array = []
count = 0
for i, lamb in enumerate(tuning_parameters_lambda):
    count += 1
    print('Tuning parameter converged at = #{c} λ {} at alpha{α}\n'.format(lamb, c=count,  α=tuning_parameters_alpha[0]))
    new_b = algorithm(X_numpy, Y_numpy, lamb, tuning_parameters_alpha[0])
    B_array.append(new_b)

dev1 = pd.DataFrame(B_array)
dev1.index=tuning_parameters_lambda
dev1.columns=['Income', 'Limit', 'Rating', 'Cards', 'Age', 'Education', 'Gender', 'Student', 'Married']
plt.plot(dev1)
plt.rcParams["figure.figsize"] = (16,10)
plt.xscale('log')
plt.xlabel('λ Tuning Params')
plt.ylabel('p=9 features')
plt.legend(loc='upper left')
plt.show()

plt.show()



