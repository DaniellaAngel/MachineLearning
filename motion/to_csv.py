import os
import csv
import random
import datetime

#V2.0motion20_15
#V2.0motion25_15

#V2.0motion20_15
#V2.0motion23_15

#V2.0resize
#V2.0motion2_0

#V2.0motion20_15
#V2.0motion20_20

#V2.0motion20_15
#V2.0motion20_17


starttime = datetime.datetime.now()

dir1 = '/home/daniella/motion_blurred/data/V2.0motion20_15'
dir2 = '/home/daniella/motion_blurred/data/V2.0motion20_20'

outdir1 ='data/V2.0motion20_15'
outdir2 = 'data/V2.0motion20_20'

fileList1 = os.listdir(dir1)
random.shuffle(fileList1)
fileList2 = os.listdir(dir2)
random.shuffle(fileList2)

fileinfo = open('V2.0motion20_15_20_20.csv','w') 

for filename1 in fileList1:
	if os.path.splitext(filename1)[1] == '.tif':
		current_name_1= os.path.join(outdir1,filename1)
		print(current_name_1)
		for filename2 in fileList2:
			if os.path.splitext(filename2)[1] == '.tif':
				current_name_2 = os.path.join(outdir2,filename2)
				if filename1==filename2:
					fileinfo.write(current_name_1+','+'0'+'\n'+current_name_2+','+'1'+'\n')
endtime = datetime.datetime.now()
interval=(endtime - starttime).seconds
print('time is',interval)
