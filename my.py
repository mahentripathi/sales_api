from sklearn.externals import joblib
import numpy as np

# Load the model that you just saved
regressor = joblib.load('model.pkl')

pred_features1=np.array([[102,2019,7,28],[102,2019,9,27]])
pred_result1=regressor.predict(pred_features1).astype('int64')

print(pred_result1)
