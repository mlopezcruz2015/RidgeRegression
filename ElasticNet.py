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

def plot_graph_deliverable_1(data, alpha):
    graph = pd.DataFrame(data)
    graph.index = tuning_parameters_lambda
    graph.columns = ['Income', 'Limit', 'Rating', 'Cards', 'Age', 'Education', 'Gender', 'Student', 'Married']
    graph.plot()

    plt.title('alpha = ' + str(alpha))
    plt.xscale('log')
    plt.xlabel('λ ')
    plt.ylabel('P Features')

    ax = plt.subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    plt.show()
    return

def center_data(df):
    temp = df - df.mean()
    return temp


def standardize_data(df):
    temp = center_data(df)
    temp = temp / temp.std()
    return temp

# Algorithm
def algorithm(X, y, lamb, alpha):

    # Randomly initialize the parameter vector
    betas = np.random.uniform(-1, 1, size=9)

    for i in range(1000):
        for k in range(len(betas)):
            Ak = np.dot(X[:, k].T, np.add(y - np.dot(X, betas), np.multiply(betas[k], X[:, k])))
            betas[k] = (np.sign(Ak) * max(0, np.abs(Ak) - (lamb * (1 - alpha) / 2)) / np.add(b[k], lamb * alpha))

    return betas

# Global Variables
X_headers = ['Income', 'Limit', 'Rating', 'Cards', 'Age', 'Education', 'Gender', 'Student', 'Married']
Y_header = 'Balance'

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
count = 0
for alpha in tuning_parameters_alpha:
    B_array = []
    for i, lamb in enumerate(tuning_parameters_lambda):
        new_b = algorithm(X_numpy, Y_numpy, lamb, alpha)
        B_array.append(new_b)

    plot_graph_deliverable_1(B_array, alpha)

plt.show()



