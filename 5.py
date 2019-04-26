import json
import datetime

base = datetime.datetime.today()

# a Python object (dict):

x = {
  "day": "Monday",
  "sales": 30,
}

y = {
  "name": "John",
  "age": 30,
}



dict_1 = {}
day1="Monday"
sales1=1223
dict_1['day']=base.strftime("%A")
dict_1['sales']=sales1

dict_2 = {}
day2="Tuesday"
sales2=1223
dict_2['day']=base.strftime("%A")
dict_2['sales']=sales1

z = [dict_1,dict_2]


# convert into JSON:
a = json.dumps(z)

# the result is a JSON string:
print(a)


#base = datetime.datetime.today()
#print(base.strftime("%A"))

