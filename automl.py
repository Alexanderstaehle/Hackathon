from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(3)
print(train.shape)
new_x = poly.fit_transform(x_train)
print(new_x.shape)

X_train, X_val, y_train, y_val = train_test_split(new_x,y_train, test_size=0.2)

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

RANDOM_SEED = 42

xgb_reg = XGBRegressor(objective='reg:squarederror', seed=RANDOM_SEED)

params = {
    'max_depth': range(4, 9),
    'n_estimators': [25, 50, 100, 200],
    'gamma': [i/10.0 for i in range(0,10)],
    'learning_rate': [0.05, 0.1, 0.5]
}

grid_search = GridSearchCV(estimator=xgb_reg, param_grid=params, cv=3, n_jobs=-1, scoring='neg_mean_squared_error')
grid_search.fit(X_train,y_train)

print("Best parameters found: ", grid_search.best_params_)
print("Lowest RMSE found: ", np.sqrt(np.abs(grid_search.best_score_)))

xgb_reg = XGBRegressor(objective='reg:squarederror', seed=RANDOM_SEED, **grid_search.best_params_)
xgb_reg.fit(X_train,y_train)

predictions = xgb_reg.predict(X_val)

rmse = np.sqrt(mean_squared_error(y_val, predictions))
print("RMSE on test set: %f" % (rmse))