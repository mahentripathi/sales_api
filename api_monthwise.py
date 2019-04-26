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
from dateutil import relativedelta
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
            print("xxxxx" + a + "yyyyy")
    
            #words = a.split(":")

            #w1 = words[1]
            #w2 = words[2]
            #w3 = words[3]
            #w4 = words[4]

            #print(words[1])
            #print(words[2])
            #print(words[3])
            #print(words[4])

            #x1=words[1].split(",")
            #userid=x1[0]
            #userid=userid[1:]
            #print("User ID:" + userid)
            #print("xxxxxx"+userid+"yyyyyy")

            #x2=words[2].split(",")
            #year=x2[0]
            #year=year[1:]
            #print("Year:" + year)
            #print("xxxxxx"+year+"yyyyyy")

            #x3=words[3].split(",")
            #month=x3[0]
            #month=month[1:]
            #print("Month:" + month)
            #print("xxxxxx"+month+"yyyyyy")

            #x4=words[4].split(",")
            #day1=x4[0]
            #x5=day1.split("}")
            #day=x5[0]
            #day=day[1:]
            #print("Day:" + day)
            #print("xxxxxx"+day+"yyyyyy")

            words = a.split(":")

            w1 = words[1]
            w2 = words[2]

            print(words[1])
            print(words[2])

            x1=words[1].split(",")
            month=x1[0]
            month=month[1:]
            print("Month:" + month)
            print("xxxxxx"+month+"yyyyyy")

            x4=words[2].split("}")
            year=x4[0]
            year=year[1:]
            print("Year:" + year)
            print("xxxxxx"+year+"yyyyyy")

            date_time_str = year + "-" + month + "-01"
            print("xxxxxx"+date_time_str+"yyyyyy")
            date_time_obj = datetime.datetime.strptime(date_time_str, '%Y-%m-%d')
            print(date_time_obj)


            
            
            base = datetime.datetime.today() 
            next_seven_days = []
            next_twelve_months = []

            #query = []

            #for x in range(0,8):
                #next_seven_days.append(base + datetime.timedelta(days=x))
 
            #for x in range(0,13):
                #next_twelve_months.append(base + relativedelta.relativedelta(months=x))

            for x in range(0,13):
                next_twelve_months.append(date_time_obj + relativedelta.relativedelta(months=x))
            

            #my_dict = json.loads(a)
            #print(my_dict)
            #print(dict['year'])

            #regressor.predict(query)

            ############################################## 
            # QUICKLY TEST QUERY HERE
            ##############################################
            pred_test=np.array([[5,2019]])
            test_res=regressor.predict(pred_test).astype('int64')

            #print("CCCCCCCCCCCCCCCC")
            #print(test_res)
            #print("CCCCCCCCCCCCCCCC")
   
            ################################################



            #userid = np.int64(userid)
            #year = np.int64(year)
            #month = np.int64(month)
            #day = np.int64(day)


            
            #pred_features0=np.array([[userid,year,month,day]])
            #pred_features1=np.array([[userid,next_seven_days[1].year,next_seven_days[1].month,next_seven_days[1].day]])
            #pred_features2=np.array([[userid,next_seven_days[2].year,next_seven_days[2].month,next_seven_days[2].day]])
            #pred_features3=np.array([[userid,next_seven_days[3].year,next_seven_days[3].month,next_seven_days[3].day]])
            #pred_features4=np.array([[userid,next_seven_days[4].year,next_seven_days[4].month,next_seven_days[4].day]])
            #pred_features5=np.array([[userid,next_seven_days[5].year,next_seven_days[5].month,next_seven_days[5].day]])
            #pred_features6=np.array([[userid,next_seven_days[6].year,next_seven_days[6].month,next_seven_days[6].day]])
            #pred_features7=np.array([[userid,next_seven_days[7].year,next_seven_days[7].month,next_seven_days[7].day]])

            
 
            #pred_result0=regressor.predict(pred_features0).astype('int64')
            #pred_result1=regressor.predict(pred_features1).astype('int64')
            #pred_result2=regressor.predict(pred_features2).astype('int64')
            #pred_result3=regressor.predict(pred_features3).astype('int64')
            #pred_result4=regressor.predict(pred_features4).astype('int64')
            #pred_result5=regressor.predict(pred_features5).astype('int64')
            #pred_result6=regressor.predict(pred_features6).astype('int64')
            #pred_result7=regressor.predict(pred_features7).astype('int64')

            pred_features0=np.array([[next_twelve_months[0].month,next_twelve_months[0].year]])
            pred_features1=np.array([[next_twelve_months[1].month,next_twelve_months[1].year]])
            pred_features2=np.array([[next_twelve_months[2].month,next_twelve_months[2].year]])
            pred_features3=np.array([[next_twelve_months[3].month,next_twelve_months[3].year]])
            pred_features4=np.array([[next_twelve_months[4].month,next_twelve_months[4].year]])
            pred_features5=np.array([[next_twelve_months[5].month,next_twelve_months[5].year]])
            pred_features6=np.array([[next_twelve_months[6].month,next_twelve_months[6].year]])
            pred_features7=np.array([[next_twelve_months[7].month,next_twelve_months[7].year]])
            pred_features8=np.array([[next_twelve_months[8].month,next_twelve_months[8].year]])
            pred_features9=np.array([[next_twelve_months[9].month,next_twelve_months[9].year]])
            pred_features10=np.array([[next_twelve_months[10].month,next_twelve_months[10].year]])
            pred_features11=np.array([[next_twelve_months[11].month,next_twelve_months[11].year]])
            pred_features12=np.array([[next_twelve_months[12].month,next_twelve_months[12].year]])



            pred_result0=regressor.predict(pred_features0).astype('int64')
            pred_result1=regressor.predict(pred_features1).astype('int64')
            pred_result2=regressor.predict(pred_features2).astype('int64')
            pred_result3=regressor.predict(pred_features3).astype('int64')
            pred_result4=regressor.predict(pred_features4).astype('int64')
            pred_result5=regressor.predict(pred_features5).astype('int64')
            pred_result6=regressor.predict(pred_features6).astype('int64')
            pred_result7=regressor.predict(pred_features7).astype('int64')
            pred_result8=regressor.predict(pred_features8).astype('int64')
            pred_result9=regressor.predict(pred_features9).astype('int64')
            pred_result10=regressor.predict(pred_features10).astype('int64')
            pred_result11=regressor.predict(pred_features11).astype('int64')
            pred_result12=regressor.predict(pred_features12).astype('int64')
 
            pred_result0 = int(pred_result0)
            pred_result1 = int(pred_result1)
            pred_result2 = int(pred_result2)
            pred_result3 = int(pred_result3)
            pred_result4 = int(pred_result4)
            pred_result5 = int(pred_result5)
            pred_result6 = int(pred_result6)
            pred_result7 = int(pred_result7)
            pred_result8 = int(pred_result8)
            pred_result9 = int(pred_result9)
            pred_result10 = int(pred_result10)
            pred_result11 = int(pred_result11)
            pred_result12 = int(pred_result12)

            #print(pred_result0)
            #print(pred_result1)
            #print(pred_result2)
            #print(pred_result3)
            #print(pred_result4)
            #print(pred_result5)
            #print(pred_result6)
            #print(pred_result7)
            #print(pred_result8)
            #print(pred_result9)
            #print(pred_result10)
            #print(pred_result11)
            #print(pred_result12)
            #print(pred_result5)
            #print(pred_result6)
            #print(pred_result7)


            #dict_0 = {}
            #dict_0['x']=next_seven_days[0].strftime("%a")
            #dict_0['value']=pred_result0


            #dict_1 = {}
            #dict_1['x']=next_seven_days[1].strftime("%a")
            #dict_1['value']=pred_result1

            #dict_2 = {}
            #dict_2['x']=next_seven_days[2].strftime("%a")
            #dict_2['value']=pred_result2

            #dict_3 = {}
            #dict_3['x']=next_seven_days[3].strftime("%a")
            #dict_3['value']=pred_result3

            #dict_4 = {}
            #dict_4['x']=next_seven_days[4].strftime("%a")
            #dict_4['value']=pred_result4


            #dict_5 = {}
            #dict_5['x']=next_seven_days[5].strftime("%a")
            #dict_5['value']=pred_result5

            #dict_6 = {}
            #dict_6['x']=next_seven_days[6].strftime("%a")
            #dict_6['value']=pred_result6

            #dict_7 = {}
            #dict_7['x']=next_seven_days[7].strftime("%a")
            #dict_7['value']=pred_result7




            dict_0 = {}
            dict_0['x']=next_twelve_months[0].strftime("%b")
            dict_0['value']=pred_result0


            dict_1 = {}
            dict_1['x']=next_twelve_months[1].strftime("%b")
            dict_1['value']=pred_result1

            dict_2 = {}
            dict_2['x']=next_twelve_months[2].strftime("%b")
            dict_2['value']=pred_result2

            dict_3 = {}
            dict_3['x']=next_twelve_months[3].strftime("%b")
            dict_3['value']=pred_result3

            dict_4 = {}
            dict_4['x']=next_twelve_months[4].strftime("%b")
            dict_4['value']=pred_result4


            dict_5 = {}
            dict_5['x']=next_twelve_months[5].strftime("%b")
            dict_5['value']=pred_result5

            dict_6 = {}
            dict_6['x']=next_twelve_months[6].strftime("%b")
            dict_6['value']=pred_result6

            dict_7 = {}
            dict_7['x']=next_twelve_months[7].strftime("%b")
            dict_7['value']=pred_result7
 
            dict_8 = {}
            dict_8['x']=next_twelve_months[8].strftime("%b")
            dict_8['value']=pred_result8

            dict_9 = {}
            dict_9['x']=next_twelve_months[9].strftime("%b")
            dict_9['value']=pred_result9

            dict_10 = {}
            dict_10['x']=next_twelve_months[10].strftime("%b")
            dict_10['value']=pred_result10


            dict_11 = {}
            dict_11['x']=next_twelve_months[11].strftime("%b")
            dict_11['value']=pred_result11

            dict_12 = {}
            dict_12['x']=next_twelve_months[12].strftime("%b")
            dict_12['value']=pred_result12


            #list_of_dicts = [dict_0,dict_1,dict_2,dict_3,dict_4,dict_5,dict_6,dict_7]
            list_of_dicts = [dict_0,dict_1,dict_2,dict_3,dict_4,dict_5,dict_6,dict_7,dict_8,dict_9,dict_10,dict_11,dict_12]

            # convert into JSON:
            json_dict = json.dumps(list_of_dicts)

            # the result is a JSON string:
            print(json_dict)
 


        
            #list = [pred_result0, pred_result1,pred_result2, pred_result3, pred_result4, pred_result5, pred_result6, pred_result7] 

            query = pd.get_dummies(pd.DataFrame(json_))
            #print(query) 
            query = query.reindex(columns=model_columns, fill_value=0)

            #print(query)

            #prediction = list(regressor.predict(query))
            #print(prediction)
            
            #return jsonify({'prediction': str(list)})
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
