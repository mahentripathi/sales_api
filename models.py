# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 17:11:49 2019

@author: hp
"""

import mysql.connector as sql
import pandas as pd
import numpy as np

#creating connection
db_connection = sql.connect(host='localhost', database='jtsboard', user='root', password='')
db_cursor = db_connection.cursor()

#fetching sinle "customer_histories table from mysql database
db_cursor.execute('SELECT id, user_id,date,customer_id FROM customer_histories')
sql_data1 = db_cursor.fetchall()

#creating dataframe for "customer_histories"
df1=pd.DataFrame(sql_data1, columns=["ch_id","user_id","date", "customer_id"])
#print(df1.head())

#converting "custmer histories" dataframe to CSV
df1.to_csv('sales_custxyz.csv', index=False)


#fetching  "note services" table from mysql database

db_cursor.execute('SELECT id, user_id, customer_id, customer_history_id, service_id, employee_id, service_price FROM note_services')
sql_data2 = db_cursor.fetchall()

# converting to "note services" dataframe
df2=pd.DataFrame(sql_data2, columns=["ns_id", "user_id", "customer_id", "ch_id", "service_id", "employee_id", "service_price"])

#converting "note services"  dataframe to csv
df2.to_csv('sales_servicexyz.csv', index=False)



#fetching  "note products" table from mysql database

db_cursor.execute('SELECT id, user_id, customer_id, customer_history_id, product_id, employee_id, sale_price FROM note_products')
sql_data3= db_cursor.fetchall()


# converting to "note products" table to  dataframe

df3=pd.DataFrame(sql_data3, columns=["np_id", "user_id", "customer_id", "ch_id", "product_id", "employee_id", "sale_price"])

#converting "note products""  dataframe to csv

df3.to_csv('sales_productxyz.csv',index=False)



#fetching  "note tickets" table from mysql database

db_cursor.execute('SELECT id, user_id, customer_id, customer_history_id, ticket_id, ticket_price,employee_id FROM note_tickets ')
sql_data4 = db_cursor.fetchall()

# converting to "note tickets" table to  dataframe

df4=pd.DataFrame(sql_data4, columns=["nt_id", "user_id", "customer_id", "ch_id","ticket_id", "ticket_price","employee_id"])

#converting "note tickets"  dataframe to csv

df4.to_csv('sales_ticketxyz.csv', index=False)



# read all converted csv file and stored in variable
a1=pd.read_csv("sales_custxyz.csv")
a2=pd.read_csv("sales_servicexyz.csv")
a3=pd.read_csv("sales_productxyz.csv")
a4=pd.read_csv("sales_ticketxyz.csv")


#merging a1 nad a2 dataset
merge1=pd.merge(a1,a2, on=['customer_id','ch_id','user_id'], how='outer')

#merging merge1 and a3 dataset
merge2=pd.merge(merge1,a3, on=['customer_id','ch_id','user_id'], how='outer')

#merging merge2 and a4 dataset
merge3=pd.merge(merge2,a4, on=['customer_id','ch_id','user_id'], how='outer')

# converting final merge data to csv
merge3.to_csv("salesxyz.csv", index=False)

#checking nan value in each column
#print(merge3.isnull().sum())

# read csv file from save place
jts_sales=pd.read_csv("salesxyz.csv")

#converting it into dataframe 
jts_sales=pd.DataFrame(jts_sales)

#fetching nan data from jts_sales
#print(jts_sales.isnull().sum())
#cust_service3=jts_sales.loc[jts_sales['date'] =='2018-08-27' ]
#features
data_clean=jts_sales.iloc[:,[1,2,7,11,14]].values

data_clean=pd.DataFrame(data_clean)


#drop nan row
#data_clean.dropna(axis=0, inplace=True)
#replace nan with '0' from date column 
#data_clean= data_clean.replace(np.nan, 0)

#drop useless row from row according to index
clean=data_clean.drop(data_clean.index[2073:2276])
#clean[0] = clean[0].apply(pd.to_datetime)
#clean=pd.DataFrame(clean)

#print all nan columns from 0,1,2,3 on date having nan
#null_columns=clean.columns[clean.isnull().any()]
#clean[null_columns].isnull().sum()
#dropp1=clean.drop[clean[0].isnull()][null_columns]

#print all nan columns from 0,1,2,3 on date having nan
clean= clean[pd.notnull(clean[0])]

#features
features=clean.iloc[:,[0,1]].values
features=pd.DataFrame(features)

"""
features=pd.to_datetime(features[0])
features= features.replace(np.nan, 0)
#convert date column to datetime and split to individuaal 'year', 'month', 'date'
features[0] = features[0].apply(pd.to_datetime)
features['year'] = [i.year for i in features[0]]
features['month'] = [i.month for i in features[0]]
features['day'] = [i.day for i in features[0]]
"""
#labels
labels=clean.iloc[:,[2,3,4]].values
labels=pd.DataFrame(labels)

#jts_labels[1] = jts_labels[1].replace(np.nan, 0)
#jts_labels=pd.DataFrame(jts_labels)
#result = pd.concat([features, labels[0]], axis=1, join='outer')
  
#labels_cleaning
#cleaning service sales column and extract numerical data
#clean 1st column and extract numerical part
#fill blank row 
labels.iloc[:,0] = labels.iloc[:,0].fillna("b''")


ls=[]
for i in  labels.iloc[:,0]:
    temp = i.split('\\')
    temp = temp[0][1:].strip("'").split(',')
    add = ''
    for j in temp:
        add += j
    if add!='':
        ls.append(float(add))
    else:
        ls.append(0.0)

labels.iloc[:,0] = ls



#clean 1st column and extract numeical part
    #get pure numerical part
labels.iloc[:,1] = labels.iloc[:,1].fillna("bytearray(b''")


ls2=[]
for i2 in  labels.iloc[:,1]:
    temp2 = i2.split('\\')
    temp2 = temp2[0][11:].strip("'").split(',')
    add2 = ''
    for j2 in temp2:
        add2 += j2
    if add2!='':
        ls2.append(add2)
    else:
        ls2.append(0.0) 
labels.iloc[:,1] = ls2

#1 st column extract [pure numerical part]
labels.iloc[:,1] = labels.iloc[:,1].str.strip().str.lower().str.replace(")","").str.replace("'", "")
#handle missing values in column1
    #repplace nan with 0
labels[1] = labels[1].replace(np.nan, 0)
#replace empty rows in 1st column with 0
labels[1] = labels[1].replace('', 0)



#extract second column with numerical part 
labels.iloc[:,2] = labels.iloc[:,2].fillna("bytearray(b")


ls3=[]
for i3 in  labels.iloc[:,2]:
    temp3 = i3.split('\\')
    temp3 = temp3[0][11:].strip("'").split(',')
    add3 = ''
    for j3 in temp3:
        add3 += j3
    if add3!='':
        ls3.append(add3)
    else:
        ls3.append(0.0)
labels.iloc[:,2] = ls3

# clean second column more with redundant word
#jts_labels.iloc[:,2] = jts_labels.iloc[:,2].apply(lambda x: x.replace(")","").replace("'",""))
labels.iloc[:,2] = labels.iloc[:,2].str.strip().str.lower().str.replace(")","").str.replace("'", "")
#handle missing values in column1
    #repplace nan with 0
labels[2] = labels[2].replace(np.nan, 0)
#replace empty rows in 1st column with 0
labels[2] = labels[2].replace('', 0)


#convert all columns '0','1','2' such columns to numeric either in 'int64' and 'float64'
#for i in range(0, len(labels.columns)):
#    labels.iloc[:,i] = pd.to_numeric(labels.iloc[:,i], errors='ignore')

#convert all columns '0','1','2' such columns to numeric either in dtype('O') 'int64' and 'float64'
cols = [0,1,2]
labels[cols] = labels[cols].applymap(np.int64)

#prediction start

#labels

#finally creates 'total_sales' column by adding such columns '0','1','2'
labels["total_sales"]=labels.sum(axis=1, skipna=True)
#df_labels.drop(["total_sales"], axis=1, inplace=True)


jtsx=features.iloc[:,[0,1]].values
#jtsx =pd.to_datetime(jtsx)
jtsx=pd.DataFrame(jtsx,columns=["u","d"])

#calculating total sales by adding service priec ticket price and sale price
jtsy=labels.loc[:,["total_sales"]].values.astype('int64')
jtsy=pd.DataFrame(jtsy, columns=['s'])

#concat two columns from dataframe
jts=pd.concat([jtsx, jtsy], axis=1)

#converting it into csv
jts.to_csv("jtstsales.csv", index=False)

#reading csv file
data = pd.read_csv('jtstsales.csv')

#fetch only nan free data
data = data[pd.notnull(data['d'])]


#features date columns
features=data.loc[:,['u','d']].values
features=pd.DataFrame(features)

#splitting date into year, month, day
features[1] = features[1].apply(pd.to_datetime)
features['year'] = [i.year for i in features[1]]
features['month'] = [i.month for i in features[1]]
features['day'] = [i.day for i in features[1]]

#features
featuresx=features.loc[:,[0,"year","month","day"]].values
featuresx=pd.DataFrame(featuresx, columns=["userid","year","month","day"])


#labels
labelsy=data.loc[:,'s'].values
labelsy=pd.DataFrame(labelsy, columns=["sales"])


#splitting dataset in training and testing dataset
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
features_train,features_test,labels_train,labels_test=train_test_split(featuresx,labelsy,test_size=0.2,random_state=0)

#features scaling
#from sklearn.preprocessing import StandardScaler
#sc=StandardScaler()
#features_train=sc.fit_transform(features_train)
#features_test=sc.transform(features_test)

#create model
#Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(features_train,labels_train)

#prediction on features_test
# Predicting the Test set results
prediction=regressor.predict(features_test).astype('int64')

df=pd.DataFrame(prediction).astype('int64')

#accuracy on training data
Score_train=regressor.score(features_train,labels_train)

#accuracy score on testing data
Score_test=regressor.score(features_test,labels_test)

#calculatig residual
#performance metrices
from sklearn import metrics
#mean_absolute_error
print('MAE:',metrics.mean_absolute_error(labels_test, prediction))
#mean_squared_error
print('MSE:',metrics.mean_squared_error(labels_test, prediction))
#mean_absolute_error
print('RMSE:',np.sqrt(metrics.mean_absolute_error(labels_test, prediction)))


#actual and predicted in dataframe

#dataframe_xy = pd.DataFrame(pred, labels_test)  
 
#claculating prediction on given data
#test
pred_features1=np.array([[33,2019,7,28],[33,2019,9,27]])
pred_result1=regressor.predict(pred_features1).astype('int64')




# Save your model
from sklearn.externals import joblib
joblib.dump(regressor, 'model.pkl')
print("Model dumped!")

# Load the model that you just saved
regressor = joblib.load('model.pkl')

# Saving the data columns from training
model_columns = list(featuresx.columns)
joblib.dump(model_columns, 'model_columns.pkl')
print("Models columns dumped!")



#prediction in loop on giving data by user and data is "date", "cust_id"
arr1 = []
elem1 = int(input("insert how many elements you want:"))
for i in range(0, elem1):
    arr1.append(int(input("Enter date :")))
    
res1=np.array([arr1])
pred1=regressor.predict(res1)

print ("jts_sales prediction on date {}: {}".format(arr1,pred1))


#infinite loop 1
    
while True:

    arr2= []
    elem2 = int(input("insert how many elements you want:"))
    for i in range(0, elem2):
        arr2.append(int(input("Enter date :")))
    
    res2=np.array([arr2])
    pred2=regressor.predict(res2)
    
    print ("jts_sales prediction on date {}:{}".format(arr2,pred2))
    
    


#save model and deploy it anywhere using flask
#method1
import pickle

#serializing our model to a file called model.pkl
pickle.dump(regressor, open("jtsdsales.pkl","wb"))


#loading a model from a file called model.pkl
with open('jtsdsales.pkl', 'rb') as handle:
    regressor = pickle.load(handle)    
# no we can call various methods over mlp_nn as as:
# Let X_test be the feature (UNIX timestamp) for which we want to predict the output 
dtsales_pred = regressor.predict(features_test).astype('int64')

#pickle

with open('model.pkl', 'rb') as handle:
    clff = pickle.load(handle)    
# no we can call various methods over mlp_nn as as:
# Let X_test be the feature (UNIX timestamp) for which we want to predict the output 
data = []
elem = int(input("insert how many elements you want:"))
for i in range(0, elem):
    data.append(int(input("Enter date and customer_id :"))) 
    
res=np.array([data])
pred_result=clff.predict(res)

print ("jts_sales prediction on date & id{}:{}".format(data,pred_result))



#Save to pickled file using joblib –
#method2

from sklearn.externals import joblib 
  # Save the model as a pickle in a file 
joblib.dump(regressor, 'regjoblib.pkl') 
  # Load the model from the file 
reg1_joblib = joblib.load('regjoblib.pkl')  
  # Use the loaded model to make predictions 
result_joblib=reg1_joblib.predict(features_test) 



#trained model
import numpy as np
pred_features1=np.array([[290,165],[270,123]])
result_joblib1=reg1_joblib.predict(pred_features1) 
print(result_joblib1)

# in for loop
import numpy as np
from sklearn.externals import joblib 
reg1_joblib = joblib.load('regjoblib.pkl')  

data = []
elem = int(input("insert how many elements you want:"))
for i in range(0, elem):
    data.append(int(input("Enter date and customer_id :"))) 
    
res=np.array([data])
pred_result=reg1_joblib.predict(res)

print ("jts_sales prediction on date & id{}:{}".format(data,pred_result))



#plotting
import matplotlib.pyplot as plt

plt.scatter(features_train[0], labels_train, color='red')

plt.plot(features_train[0], regressor.predict(features_train), color='blue')
plt.title("sales_prediction for training")
plt.xlabel("data")
plt.ylabel("sales")

plt.show()




import matplotlib.pyplot as plt
import seaborn as sns

rp = sns.regplot(x=features_test[0], y=pred)
rp = sns.regplot(x=features_test[1], y=pred)


#convert datetime to individual column with 'year', 'month', 'date'



#data['date'] = data['date'].apply(pd.to_datetime)
#data['year'] = [i.year for i in data['date']]
#data['month'] = [i.month for i in data['date']]
#data['day'] = [i.day for i in data['date']]
    

