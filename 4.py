#datetime_str = '09/19/18'
#object = datetime.strptime(datetime_str, '%m/%d/%y')
#print(object)

userid=32

next_seven_dates = []

import datetime
base = datetime.datetime.today()

next_seven_days = []
query = []

for x in range(0,7):
      next_seven_days.append(base + datetime.timedelta(days=x))

#print(next_seven_days)

print(next_seven_days[0].year)


#for y in range(0,7):
#	print(userid, next_seven_days[y].year, next_seven_days[y].month, next_seven_days[y].day)






