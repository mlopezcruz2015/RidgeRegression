import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def center_data(df):
    temp = df - df.mean()
    return temp


def standardize_data(df):
    temp = center_data(df)
    temp = temp / temp.std()
    return temp

def RidgeRegression(x_array, y_array, b_array, tuning_parameter, learning_rate, iterations):

    for k in range(iterations):

        u = np.exp(np.matmul(x_array, b_array))

        p = np.zeros((u.shape[0], u.shape[1]))
        p = u/u.sum(axis=0)

        z = np.zeros((11, 5))
        for a in range(5):
            z[0][a] = b_array[0][a]

        b_array = b_array + learning_rate * (np.dot(np.transpose(x_array), (y_array-p)) - (2 * tuning_parameter * (b_array-z)))

    return b_array

def cross_validation(dataframe, folds, learning_rate, tuning_parameter):

    shuffled_data = dataframe.sample(frac=1).reset_index(drop=True)
    CCE = 0

    y_temp = shuffled_data[Y_header]

    Y_mapped = y_temp.map({'African': 1, 'European': 2, 'EastAsian': 3, 'Oceanian': 4, 'NativeAmerican': 5})
    Y_mapped = Y_mapped.map({1: np.array([1, 0, 0, 0, 0]), 2: np.array([0, 1, 0, 0, 0]), 3: np.array([0, 0, 1, 0, 0]),
                             4: np.array([0, 0, 0, 1, 0]), 5: np.array([0, 0, 0, 0, 1])})

    Y_mapped = Y_mapped.to_numpy()

    # Create Y
    y = np.zeros((Y_mapped.shape[0], 5))

    for i in range(Y_mapped.shape[0]):
        for j in range(5):
            y[i][j] = Y_mapped[i][j]

    y = y - y.mean()

    # Set up Data
    validation_sets = [shuffled_data[:36],
                       shuffled_data[36:72],
                       shuffled_data[72:109],
                       shuffled_data[109:146],
                       shuffled_data[146:184]]

    training_sets = []
    training_sets.append(pd.concat([shuffled_data, validation_sets[0]]).drop_duplicates(keep=False))
    training_sets.append(pd.concat([shuffled_data, validation_sets[1]]).drop_duplicates(keep=False))
    training_sets.append(pd.concat([shuffled_data, validation_sets[2]]).drop_duplicates(keep=False))
    training_sets.append(pd.concat([shuffled_data, validation_sets[3]]).drop_duplicates(keep=False))
    training_sets.append(pd.concat([shuffled_data, validation_sets[4]]).drop_duplicates(keep=False))


    Y_test = [y[:36],y[36:72],y[72:109],y[109:146],y[146:184]]
    Y_train = [y[36:184],
               np.concatenate((y[:36], y[72:184])),
               np.concatenate((y[:72], y[109:184])),
               np.concatenate((y[:109], y[146:184])),
               y[:146]]

    for k in range(5):

        # X
        X_train = training_sets[k][X_headers]
        X_train = standardize_data(X_train)
        X_train = np.append(np.ones((X_train.shape[0], 1)), X_train, 1)

        X_test = validation_sets[k][X_headers]
        X_test = np.append(np.ones((X_test.shape[0], 1)), X_test, 1)

        # Y
        Y_test_i = Y_test[k]
        Y_train_i = Y_train[k]

        B = np.zeros((11, 5))
        B = b.astype(float)

        B = RidgeRegression(X_train, Y_train_i, B, tuning_parameter, learning_rate, 10000)

        u = np.exp(np.matmul(X_test, B))

        p = np.zeros((u.shape[0], u.shape[1]))
        for i in range(p.shape[0]):
            for j in range(p.shape[1]):
                p[i][j] = u[i][j] / (np.sum(u[[i]]))

        CCE += -1/5 * np.sum(Y_test_i[k] * np.log10(p))

    return CCE


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

y_temp = dataframe[Y_header]

# Center Data
dataframe = dataframe - dataframe.mean()

x = dataframe[X_headers]

# Standardize Data
x = x.to_numpy()
x = x/x.std()

# Add column of 1's
x = np.append(np.ones((183, 1)), x, 1)

# Encode Ancestries
y_temp = y_temp.map({'African':1, 'European':2, 'EastAsian':3, 'Oceanian':4, 'NativeAmerican':5})
y_mapped = y_temp.map({1:np.array([1,0,0,0,0]), 2:np.array([0,1,0,0,0]), 3:np.array([0,0,1,0,0]), 4:np.array([0,0,0,1,0]), 5:np.array([0,0,0,0,1])})
y_mapped = y_mapped.to_numpy()
print(y_mapped)

# Create Y
y = np.zeros((183, 5))

for i in range(183):
    for j in range(5):
        y[i][j] = y_mapped[i][j]


# Deliverable 1

b_array = []

for l in range(len(tuning_parameters_lambda)):

    b = np.zeros((11, 5))
    b = b.astype(float)

    b = RidgeRegression(x, y, b, tuning_parameters_lambda[l], learningRateA, 1)

    b_array.append(b)

strings = ["African", "European", "East Asian", "Oceanian", "Native American"]

plt.figure(figsize=(25,18))

for i in range(5):
    pc1 = []
    pc2 = []
    pc3 = []
    pc4 = []
    pc5 = []
    pc6 = []
    pc7 = []
    pc8 = []
    pc9 = []
    pc10 = []

    for k in range(len(tuning_parameters_lambda)):
        pc1.append(b_array[k][1, i])
        pc2.append(b_array[k][2, i])
        pc3.append(b_array[k][3, i])
        pc4.append(b_array[k][4, i])
        pc5.append(b_array[k][5, i])
        pc6.append(b_array[k][6, i])
        pc7.append(b_array[k][7, i])
        pc8.append(b_array[k][8, i])
        pc9.append(b_array[k][9, i])
        pc10.append(b_array[k][10, i])

    plt.subplot(2, 3, i+1)
    title = strings[i]
    plt.title(title)
    plt.xlabel("Lambda (10^x)")
    plt.ylabel("Standardized Coefficients (Beta)")
    plt.plot(np.log10(tuning_parameters_lambda), pc1, label="PC-1")
    plt.plot(np.log10(tuning_parameters_lambda), pc2, label="PC-2")
    plt.plot(np.log10(tuning_parameters_lambda), pc3, label="PC-3")
    plt.plot(np.log10(tuning_parameters_lambda), pc4, label="PC-4")
    plt.plot(np.log10(tuning_parameters_lambda), pc5, label="PC-5")
    plt.plot(np.log10(tuning_parameters_lambda), pc6, label="PC-6")
    plt.plot(np.log10(tuning_parameters_lambda), pc7, label="PC-7")
    plt.plot(np.log10(tuning_parameters_lambda), pc8, label="PC-8")
    plt.plot(np.log10(tuning_parameters_lambda), pc9, label="PC-9")
    plt.plot(np.log10(tuning_parameters_lambda), pc10, label="PC-10")

    # plt.plot(pd.DataFrame(b_array[0], index=exponents))
    plt.legend()

# plt.show()

# End Deliverable 1

b_array = []

dataframe = pd.read_csv('/Users/mlopezcruz2015/Documents/AIProgrammingAssignments/TrainingData_N183_p10.csv', ',',
                        usecols=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10', 'Ancestry'])
folds = 5
CCE_folds = []

smallest_CV = 0
optimal_lamb = 0

for lamb in tuning_parameters_lambda:
    CCE = cross_validation(dataframe, folds, learningRateA, lamb)

    #Deliverable 3
    if smallest_CV == 0 or CCE < smallest_CV:
        smallest_CV = CCE
        optimal_lamb = lamb

    CCE_folds.append(CCE)

print(optimal_lamb)

# Plot
plt.clf()
plt.xlabel("Lambda (10^x)")
plt.ylabel("CCE (CV5 error)")
plt.plot(pd.DataFrame(CCE_folds, index=exponents))
plt.show()

print('done')
