import json

#no = '{"userid": 102, "year": 2019, "month": 5, "day": 25}'

#no = {'userid': 102, 'year': 2019, 'month': 5, 'day': 25}

#print(no)

#n = '"' + no + '"'

#print(no)



#person_dict = json.loads(x)
#print(person_dict)
#print(person_dict['year'])

n = "{'userid': 102, 'year': 2019, 'month': 5, 'day': 25}"
print(n)
y = str(n).replace("'", '"')
print(y)


z = "{"userid": 102, "year": 2019, "month": 5, "day": 25}"
z = z[1:-1]

z = "'" + z + "'"
print(z)

#dict = json.loads(z)
#print(dict)
#print(dict['year'])




