import os
import csv
import random
import datetime
#V2.0resize============0
#V2.0motion20_20=======1
#V2.0motion20_30=======2
#V2.0motion20_45=======3


#V2.0resize============0
#V2.0motion20_30=======1
#V2.0motion40_30=======2
#V2.0motion60_30=======3


starttime = datetime.datetime.now()

# dir1 = '/home/daniella/motion_blurred/data/V2.0resize'
# dir2 = '/home/daniella/motion_blurred/data/V2.0motion20_15'
# dir3 = '/home/daniella/motion_blurred/data/V2.0motion20_30'
# dir4 = '/home/daniella/motion_blurred/data/V2.0motion20_45'
# outdir1 ='data/V2.0resize'
# outdir2 = 'data/V2.0motion20_15'
# outdir3 = 'data/V2.0motion20_30'
# outdir4 = 'data/V2.0motion20_45'


dir1 = '/home/daniella/motion_blurred/data/V2.0resize'
dir2 = '/home/daniella/motion_blurred/data/V2.0motion20_30'
dir3 = '/home/daniella/motion_blurred/data/V2.0motion40_30'
dir4 = '/home/daniella/motion_blurred/data/V2.0motion60_30'
outdir1 ='data/V2.0resize'
outdir2 = 'data/V2.0motion20_30'
outdir3 = 'data/V2.0motion40_30'
outdir4 = 'data/V2.0motion60_30'


fileList1 = os.listdir(dir1)
# random.shuffle(fileList1)
fileList2 = os.listdir(dir2)
# random.shuffle(fileList2)
fileList3 = os.listdir(dir3)
# random.shuffle(fileList3)
fileList4 = os.listdir(dir4)
# random.shuffle(fileList4)

fileinfo1 = open('V2.0resize.csv','w')
fileinfo2 = open('V2.0motion20_30_len.csv','w')
fileinfo3 = open('V2.0motion40_30_len.csv','w')
fileinfo4 = open('V2.0motion60_30_len.csv','w') 

# fileinfo1 = open('V2.0resize.csv','w')
# fileinfo2 = open('V2.0motion20_15.csv','w')
# fileinfo3 = open('V2.0motion20_30.csv','w')
# fileinfo4 = open('V2.0motion20_45.csv','w') 

for filename1 in fileList1:
		current_name_1= os.path.join(outdir1,filename1)
		print current_name_1
		fileinfo1.write(current_name_1+','+'0'+'\n')
for filename2 in fileList2:
		current_name_2= os.path.join(outdir2,filename2)
		print current_name_2
		fileinfo2.write(current_name_2+','+'1'+'\n')
for filename3 in fileList3:
		current_name_3= os.path.join(outdir3,filename3)
		print current_name_3
		fileinfo3.write(current_name_3+','+'2'+'\n')
for filename4 in fileList4:
		current_name_4= os.path.join(outdir4,filename4)
		print current_name_4
		fileinfo4.write(current_name_4+','+'3'+'\n')

endtime = datetime.datetime.now()
interval=(endtime - starttime).seconds
print('time is',interval)
