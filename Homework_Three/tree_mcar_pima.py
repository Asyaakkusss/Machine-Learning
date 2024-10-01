'''
A simple model for missing data is to assume that entries are missing completely at random
(MCAR). In other words, each entry 𝑥𝑗
(𝑖) of the feature matrix 𝑿 is missing with some probability
𝑝. The provided function
generate_mcar_data(data, missing_rate, missing_value=np.nan,
random_state=1)
randomly chooses entries from data to be missing at the provided missing data rate, replacing
them with NaN values. The seed for the random number generator used to choose missing entries
can be set using random_state.
Use generate_mcar_data() to generate missing entries in the Pima Indians data at a missing
rate of 20% with random_state=1 to provide a fixed seed. Then, split the data 50/50 into training
and test sets using train_test_split(), also with random_state=1.
Choose an approach to impute the missing data and then train a decision tree. Try out different
approaches for missing data imputation and different hyperparameters for the decision tree to see
their effects on the test set classification accuracy. Report the highest test set accuracy you are
able to obtain. Submit your code in a file named tree_mcar_pima.py.
'''

import numpy as np

def generate_mcar_data(data, missing_rate, missing_value=np.nan,
random_state=1):
    # Generate uniform random numbers and choose entries smaller than   
    # missing_rate to be missing using a mask
    rng = np.random.default_rng(random_state)
    random_missing = rng.random((data.shape[0], data.shape[1] - 1))
    mask = np.where(random_missing < missing_rate, 1, 0)
    mask_with_label = np.hstack((mask, np.zeros((data.shape[0], 1))))
    data_missing = data.copy()
    data_missing[mask_with_label == 1] = missing_value
    return data_missing