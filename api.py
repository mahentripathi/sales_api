# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 16:55:43 2019

@author: hp
"""



# Dependencies
from flask import Flask, request, jsonify
from sklearn.externals import joblib
import traceback
import pandas as pd
import numpy as np
import sys
import json

################
import datetime
###############

# Your API definition
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if regressor:
        try:
            json_ = request.json
            print(json_)
            m = str(json_).strip('[]')
            #print(m) 
            n = '"' + m + '"'  
            #print(n)
            y = str(n).replace("'", '"')
            #print("HHHHHHH") 
            #print(y)
            #print("IIIIII")
            z = y[1:-1]
            a = "'" + z + "'"
            #print("xxxxx" + a + "yyyyy")
    
            words = a.split(":")

            w1 = words[1]
            w2 = words[2]
            w3 = words[3]
            w4 = words[4]

            #print(words[1])
            #print(words[2])
            #print(words[3])
            #print(words[4])

            x1=words[1].split(",")
            userid=x1[0]
            userid=userid[1:]
            print("User ID:" + userid)
            #print("xxxxxx"+userid+"yyyyyy")

            x2=words[2].split(",")
            year=x2[0]
            year=year[1:]
            print("Year:" + year)
            #print("xxxxxx"+year+"yyyyyy")

            x3=words[3].split(",")
            month=x3[0]
            month=month[1:]
            print("Month:" + month)
            #print("xxxxxx"+month+"yyyyyy")

            x4=words[4].split(",")
            day1=x4[0]
            x5=day1.split("}")
            day=x5[0]
            day=day[1:]
            print("Day:" + day)
            #print("xxxxxx"+day+"yyyyyy")
            
            base = datetime.datetime.today() 
            next_seven_days = []
            #query = []

            for x in range(0,8):
                next_seven_days.append(base + datetime.timedelta(days=x))

            

            #dict = json.loads(a)
            #print(dict)
            #print(dict['year'])

            #regressor.predict(query)

            #pred_features1=np.array([[102,2019,5,25]])

            userid = np.int64(userid)
            year = np.int64(year)
            month = np.int64(month)
            day = np.int64(day)

            pred_features0=np.array([[userid,year,month,day]])
            pred_features1=np.array([[userid,next_seven_days[1].year,next_seven_days[1].month,next_seven_days[1].day]])
            #pred_features2=np.array([[userid,next_seven_days[2].year,next_seven_days[2].month,next_seven_days[2].day]])
            #pred_features3=np.array([[userid,next_seven_days[3].year,next_seven_days[3].month,next_seven_days[3].day]])
            #pred_features4=np.array([[userid,next_seven_days[4].year,next_seven_days[4].month,next_seven_days[4].day]])
            #pred_features5=np.array([[userid,next_seven_days[5].year,next_seven_days[5].month,next_seven_days[5].day]])
            #pred_features6=np.array([[userid,next_seven_days[6].year,next_seven_days[6].month,next_seven_days[6].day]])
            #pred_features7=np.array([[userid,next_seven_days[7].year,next_seven_days[7].month,next_seven_days[7].day]])

 
            pred_result0=regressor.predict(pred_features0).astype('int64')
            pred_result1=regressor.predict(pred_features1).astype('int64')
            pred_result2=regressor.predict(pred_features0).astype('int64')
            pred_result3=regressor.predict(pred_features1).astype('int64')
            pred_result4=regressor.predict(pred_features0).astype('int64')
            pred_result5=regressor.predict(pred_features1).astype('int64')
            pred_result6=regressor.predict(pred_features0).astype('int64')
            pred_result7=regressor.predict(pred_features1).astype('int64')
 
            print("HHHHHHHHHHHHHHHHHHHH") 
            print(type(pred_result0))
            print(type(pred_result1))
            print("IIIIIIIIIIIIIIIIIIII")

            pred_result0 = int(pred_result0)
            pred_result1 = int(pred_result1)
            pred_result2 = int(pred_result2)
            pred_result3 = int(pred_result3)
            pred_result4 = int(pred_result4)
            pred_result5 = int(pred_result5)
            pred_result6 = int(pred_result6)
            pred_result7 = pred_result7.tolist()
            
            print("HHHHHHHHHHHHHHHHHHHH")
            print(type(pred_result0))
            print(type(pred_result1))
            print("IIIIIIIIIIIIIIIIIIII")


            #print(pred_result2)
            #print(pred_result3)
            #print(pred_result4)
            #print(pred_result5)
            #print(pred_result6)
            #print(pred_result7)

            dict_0 = {}
            dict_0['day']=next_seven_days[0].strftime("%A")
            dict_0['sales']=pred_result0


            dict_1 = {}
            dict_1['day']=next_seven_days[1].strftime("%A")
            dict_1['sales']=pred_result1

            dict_2 = {}
            dict_2['day']=next_seven_days[2].strftime("%A")
            dict_0['sales']=pred_result2

            dict_3 = {}
            dict_3['day']=next_seven_days[3].strftime("%A")
            dict_3['sales']=pred_result3

            dict_4 = {}
            dict_4['day']=next_seven_days[4].strftime("%A")
            dict_4['sales']=pred_result4


            dict_5 = {}
            dict_5['day']=next_seven_days[5].strftime("%A")
            dict_5['sales']=pred_result5

            dict_6 = {}
            dict_6['day']=next_seven_days[6].strftime("%A")
            dict_6['sales']=pred_result6

            #dict_7 = {}
            #dict_7['day']=next_seven_days[7].strftime("%A")
            #dict_7['sales']=pred_result7



            #list_of_dicts = [dict_0,dict_1,dict_2,dict_3,dict_4,dict_5,dict_6]
 
            list_of_dicts = [dict_0,dict_1]


            # convert into JSON:
            #json_dict = json.dumps(list_of_dicts)

            # the result is a JSON string:
            #print(json_dict)
 


        
            list = [pred_result0, pred_result1,pred_result2, pred_result3, pred_result4, pred_result5, pred_result6, pred_result7] 


            query = pd.get_dummies(pd.DataFrame(json_))
            #print(query) 
            query = query.reindex(columns=model_columns, fill_value=0)

            #print(query)

            #prediction = list(regressor.predict(query))
            #print("HHHHHHHH")
            #print(prediction)
            #print("IIIIIIIII")
            
            return jsonify({'prediction': str(list)})
            #return jsonify({'prediction': str(prediction)})
            return json_dict
 

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 12345 # If you don't provide any port the port will be set to 12345

    regressor = joblib.load("model.pkl") # Load "model.pkl"
    print ('Model loaded')
    model_columns = joblib.load("model_columns.pkl") # Load "model_columns.pkl"
    print ('Model columns loaded')
    print(model_columns)



    app.run(port=port, debug=True)
