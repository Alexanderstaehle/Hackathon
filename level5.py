import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

RANDOM_SEED = 675
print("Random seed: ", RANDOM_SEED)

train = pd.read_csv('data/train_data.csv')
test = pd.read_csv('data/test_data.csv')
print(train.shape)
print(train.head())

# remove nans from train
train = train.dropna(axis=0, how='any')
print(train.describe())
print(train.shape)
train = train[(train["AMPS"] <= 1) & (train["AMPS"] >= 0)]
print(train.shape)
train = train.reset_index(drop=True)
# convert unknown to C or K
for i in range(len(train)):
    if train.iloc[i]['UNIT'] == '?':
        if train.iloc[i]['TEMP'] > -98 and train.iloc[i]['TEMP'] < 102:
            train.at[i, 'UNIT'] = train.at[i, 'UNIT'].replace('?', 'C')
        else:
            train.at[i, 'UNIT'] = train.at[i, 'UNIT'].replace('?', 'K')

train.loc[train["UNIT"] == "K", "TEMP"] = train.loc[train["UNIT"] == "K", "TEMP"] - 273.15
train.loc[train["UNIT"] == "K", "UNIT"] = "C"

# mean of TEMP where UNIT is C
print(round(train.loc[train["UNIT"] == "C", "TEMP"].mean(), 2))
print(round(train.loc[train["UNIT"] == "C", "TEMP"].std(), 0))

train = train.drop(['UNIT'], axis=1)
test = test.drop(['UNIT'], axis=1)

# One hot encoding
# Train
ohe = OneHotEncoder(handle_unknown='ignore')
encoded_cols = ohe.fit_transform(train[['MODE', 'POWER']]).toarray()
encoded_df = pd.DataFrame(encoded_cols, columns=ohe.get_feature_names(['MODE', 'POWER']))
train = pd.concat([train, encoded_df], axis=1)
train = train.drop(['MODE', 'POWER'], axis=1)
# Test
encoded_cols = ohe.transform(test[['MODE', 'POWER']]).toarray()
encoded_df = pd.DataFrame(encoded_cols, columns=ohe.get_feature_names(['MODE', 'POWER']))
test = pd.concat([test, encoded_df], axis=1)
test = test.drop(['MODE', 'POWER'], axis=1)
print("tes")
x_val = test
run_train = train

# split train into train and val using train_test_split
train, test = train_test_split(train, test_size=0.2, random_state=RANDOM_SEED)
train, test = train_test_split(train, test_size=0.2, random_state=RANDOM_SEED)

# split train into x and y take the OUTPUT column as y
y_train = train['OUTPUT']
x_train = train.drop(['OUTPUT'], axis=1)
y_test = test['OUTPUT']
x_test = test.drop(['OUTPUT'], axis=1)
y_train_run = run_train['OUTPUT']
x_train_run = run_train.drop(['OUTPUT'], axis=1)

print(x_train.shape)

# define XGBoost regressor
xgb_reg = XGBRegressor(objective='reg:squarederror', seed=RANDOM_SEED)

params = {
    'max_depth': range(4, 9),
    'n_estimators': [50, 100, 200],
    'gamma': [i / 10.0 for i in range(0, 10)],
    'learning_rate': [0.05, 0.1, 0.5],
    'subsample': [i / 10.0 for i in range(6, 10)],
    'colsample_bytree': [i / 10.0 for i in range(6, 10)],
}

grid_search = GridSearchCV(estimator=xgb_reg, param_grid=params, cv=3, n_jobs=-1, scoring='neg_mean_squared_error')
grid_search.fit(x_train, y_train)

print("Best parameters found: ", grid_search.best_params_)
print("Lowest RMSE found: ", np.sqrt(np.abs(grid_search.best_score_)))

xgb_reg = XGBRegressor(objective='reg:squarederror', seed=RANDOM_SEED, **grid_search.best_params_)
xgb_reg.fit(x_train_run, y_train_run)

y_hat = xgb_reg.predict(x_test)

rmse = np.sqrt(mean_squared_error(y_test, y_hat))
print("RMSE on test set: %f" % (rmse))

# yhat = xgb_reg.predict(x_val)
# # write y_hat to a txt file with one entry per line
# with open('data/y_hat.txt', 'w') as f:
#     for item in yhat:
#         f.write("%s \n" % item)
