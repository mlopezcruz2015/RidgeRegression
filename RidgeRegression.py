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

# Global Variables
X_headers = ['Income', 'Limit', 'Rating', 'Cards', 'Age', 'Education', 'Gender', 'Student', 'Married']
Y_header = 'Balance'
learningRateA = 10**-5
tuning_parameters = []
exponents = np.linspace(-2, 4, 7)
for n in exponents:
    tuning_parameters.append(10**n)


def center_data(df):
    temp = df - df.mean()
    return temp


def standardize_data(df):
    temp = center_data(df)
    temp = temp / temp.std()
    return temp


# Algorithm 1 - Ridge Regression
def ridge_regression(X, Y, learning_rate, tuning_parameter):
    # Randomly initialize the parameter vector
    betas = np.random.uniform(-1, 1, size=9)

    for i in range(10**5):
        previous_betas = betas
        betas = betas - ((2 * learning_rate) * ((betas * tuning_parameter) - np.matmul(np.transpose(X), (Y - np.matmul(X, betas)))))
        if np.array_equal(previous_betas, betas):
            break

    return betas


def cross_validation(dataframe, folds, learning_rate, tuning_parameter):
    shuffled_data = dataframe.sample(frac=1).reset_index(drop=True)
    split_array = np.array_split(shuffled_data,folds)
    MSE = 0

    for i in range(folds):
        err = 0

        validation_set = split_array[i]
        training_set = pd.concat([shuffled_data, validation_set]).drop_duplicates(keep=False)

        X_train = training_set[X_headers]
        X_train = standardize_data(X_train).to_numpy()

        Y_train = training_set[Y_header]
        Y_train = center_data(Y_train).to_numpy()

        X_test = validation_set[X_headers]
        X_test = standardize_data(X_test).to_numpy()

        Y_test = validation_set[Y_header]
        Y_test = center_data(Y_test).to_numpy()

        B = ridge_regression(X_train, Y_train, learning_rate, tuning_parameter)

        n = len(Y_test)
        for x in range(n):
            err += (Y_test[x] - np.dot(X_test[x], B)) ** 2
        MSE = MSE + err / n

    MSE = MSE / 5
    return MSE


# Centering and Scaling X (Standardizing)
X_numpy = dataframe[X_headers]
X_numpy = standardize_data(X_numpy).to_numpy()

# Centering Y
Y_numpy = dataframe[Y_header]
Y_numpy = center_data(Y_numpy).to_numpy()

B_array = []

# Ridge Regression
for alpha in tuning_parameters:
    B = ridge_regression(X_numpy, Y_numpy, learningRateA, alpha)
    B_array.append(B)

# Plot
plt.xlabel("Lambda (10^x)")
plt.ylabel("Standardized Coefficients (Beta)")
plt.plot(pd.DataFrame(B_array, index=exponents))
plt.show()

folds = 5
MSE_folds = []

smallest_CV = 0
optimal_alpha = 0

for alpha in tuning_parameters:
    MSE = cross_validation(dataframe, folds, learningRateA, alpha)

    #Deliverable 3
    if smallest_CV == 0 or MSE < smallest_CV:
        smallest_CV = MSE
        optimal_alpha = alpha

    MSE_folds.append(MSE)

print(optimal_alpha)

# Plot
plt.xlabel("Lambda (10^x)")
plt.ylabel("MSE (CV5 error)")
plt.plot(pd.DataFrame(MSE_folds, index=exponents))
plt.show()

# Deliverable 4
B = ridge_regression(X_numpy, Y_numpy, learningRateA, optimal_alpha)
print(B)


if __name__ == '__main__':
    print("Program Complete")
