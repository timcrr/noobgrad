import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
np.random.seed(1234)
plt.close("all")

house = pd.read_csv("/Users/timcr/fun/noobgrad/data/house/train.csv", delimiter=",")
house_test = pd.read_csv("/Users/timcr/fun/noobgrad/data/house/test.csv", delimiter=",")
X = house[["LotArea", "OverallQual", "OverallCond", "YearBuilt", "GrLivArea", "BedroomAbvGr", "TotRmsAbvGrd", "GarageArea"]]
X_test = house_test[["LotArea", "OverallQual", "OverallCond", "YearBuilt", "GrLivArea", "BedroomAbvGr", "TotRmsAbvGrd", "GarageArea"]]
X_test = X_test.fillna(value=0)
y = house["SalePrice"]
y_log = np.log1p(y)
scaler = StandardScaler()
X_train, X_val, y_train, y_val = train_test_split(X, y_log, test_size=0.2, random_state=1234)
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
X_val = scaler.fit_transform(X_val)

def validate(model):
  y_pred = model.predict(X_test)
  y_pred = np.expm1(y_pred)
  submission = pd.DataFrame({"Id": house_test["Id"], "SalePrice": y_pred})
  submission.to_csv("results/housep_sgdreg.csv", index=False)

def train_step(model):
  batch_size = 128
  n_epochs = 50
  losses = []
  for epoch in range(n_epochs):
    indices = np.random.permutation(len(X_train))
    X_train = X_train[indices]
    y_train = y_train.iloc[indices]
    for start in range(0, len(X_train), batch_size):
      end = start + batch_size
      batch_X = X_train[start:end]
      batch_y = y_train[start:end]
      model.partial_fit(batch_X, batch_y)
    y_pred_log = model.predict(X_val)
    log_rmse = root_mean_squared_error(y_val, y_pred_log)
    losses.append(log_rmse)
    print(f"epoch {epoch + 1}, rmse= {log_rmse:.4f}")

if __name__ == "__main__":
  model = SGDRegressor(learning_rate='constant', eta0=1e-4, max_iter=50, penalty="l2", alpha=0.0001)
  model.fit(X_train, y_train)
  y_pred = model.predict(X_val)
  log_rmse = root_mean_squared_error(y_val, y_pred)
  print(f"log rmse: {log_rmse:.4f}")
  validate(model)
  
  '''
  y_pred_log = model.predict(X_val)
  y_pred = np.expm1(y_pred_log)
  plt.figure(figsize=(8,6))
  plt.scatter(y_val, y_pred_log, alpha=0.5)
  min_val = min(y_val.min(), y_pred_log.min())
  max_val = max(y_val.max(), y_pred_log.max())
  plt.plot([min_val, max_val], [min_val, max_val], 'b--')
  plt.show()
  '''
