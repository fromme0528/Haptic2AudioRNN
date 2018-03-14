# Devide acceleraion file from android application
# Application : AcceleraionExplorer
# 50Hz
# Compensate some blank

# D\divide_accel.py


8배복사
ㄱㄱ


import os
import csv
with open(os.path.join("acceleration_0213\wood_hit.csv"),'r') as csvfile:
	data_accel = csv.reader(csvfile)
	tmp = 40
	blankCatch = [0,0,0,0]
	result = list()
	for idx, data in enumerate(data_accel):
		if idx<tmp:
			continue
		if idx>tmp+24:
			with open (os.path.join("input_accel_0213\wood_accel_"+str(int((idx+10)/50))+'.csv'),'w',newline='') as fs:
				wr = csv.writer(fs)
				for row in result:
					wr.writerow(row)
			result = []
			tmp = tmp+50
			#print(str(int((idx+10)/50)))
			continue
		# 손으로 보정
		# 연속으로 비어있는 칸도 있고 그럼.
		
		for i in range(1,4):

			if data[i] == "":
				print(idx)
		new = data[1:]
		new.append('0.0')
		
		result.append(new)
		#result.append(data[1:])



'''
with open(os.path.join("acceleration_0213\water_hit.csv"),'r') as csvfile:
	data_accel = csv.reader(csvfile)
	tmp = 15
	blankCatch = [0,0,0,0]
	result = list()
	for idx, data in enumerate(data_accel):
		if idx<tmp:
			continue
		if idx>tmp+24:
			with open (os.path.join("input_accel_0213\water_accel_"+str(int((idx+10)/50))+'.csv'),'w',newline='') as fs:
				wr = csv.writer(fs)
				for row in result:
					wr.writerow(row)
			result = []
			tmp = tmp+50
			print(str(int((idx+10)/50)))
			continue
		# 손으로 보정
		# 연속으로 비어있는 칸도 있고 그럼.
		for i in range(1,4):
			if data[i] == '':
				print(idx)
		new = data[1:]
		new.append('1.0')
		
		result.append(new)
		#result.append(data[1:])

'''
'''
# Devide acceleraion file from android application
# Application : AcceleraionExplorer
# 50Hz
# Compensate some blank

import os
import csv

with open(os.path.join("acceleraion_0213\wood_hit.csv"),'r') as csvfile:
	data_accel = csv.reader(csvfile)
	tmp = 15
	
	result = list()
	for idx, data in enumerate(data_accel):
		if idx<tmp:
			continue
		if idx>tmp+24:
			with open (os.path.join("input_accel\sample_accel_"+str(int((idx+10)/50))+'.csv'),'w',newline='') as fs:
				wr = csv.writer(fs)
				for row in result:
					wr.writerow(row)
			result = []
			tmp = tmp+50
			print(idx)
			continue
		result.append(data[1:])
		#		result.append(data[1:])

		'''