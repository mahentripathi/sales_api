from sklearn.externals import joblib

model_columns = ['userid', 'year', 'month', 'day', 'mode']
joblib.dump(model_columns, 'model_columns1.pkl')
