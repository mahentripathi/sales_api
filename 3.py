import json

n = '{"userid": 102, "year": 2019, "month": 5, "day": 25}'
#print(n)

words = n.split(":")

w1 = words[1]
w2 = words[2]
w3 = words[3]
w4 = words[4]

print(words[1])
print(words[2])
print(words[3])
print(words[4])

x1=words[1].split(",")
userid=x1[0]
userid=userid[1:]
print("User ID:" + userid)
print("xxxxxx"+userid+"yyyyyy")

x2=words[2].split(",")
year=x2[0]
year=year[1:]
print("Year:" + year)
print("xxxxxx"+year+"yyyyyy")

x3=words[3].split(",")
month=x3[0]
month=month[1:]
print("Month:" + month)
print("xxxxxx"+month+"yyyyyy")

x4=words[4].split(",")
day1=x4[0]
x5=day1.split("}")
day=x5[0]
day=day[1:]
print("Day:" + day)
print("xxxxxx"+day+"yyyyyy")

#dict = json.loads(n)
#print(dict)
#print(dict['year'])
