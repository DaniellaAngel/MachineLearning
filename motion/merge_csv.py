import csv
import heapq
# csvfile1 = file('V2.0resize.csv','rb')
# csvfile2 = file('V2.0motion20_30_len_pred.csv','rb')
# csvfile3 = file('V2.0motion40_30_len_pred.csv','rb')
# csvfile4 = file('V2.0motion60_30_len_pred.csv','rb')

csvfile1 = file('V2.0resize.csv','rb')
csvfile2 = file('V2.0motion20_15_pred.csv','rb')
csvfile3 = file('V2.0motion20_30_pred.csv','rb')
csvfile4 = file('V2.0motion20_45_pred.csv','rb')

reader1 = csv.reader(csvfile1)
reader2 = csv.reader(csvfile2)
reader3 = csv.reader(csvfile3)
reader4 = csv.reader(csvfile4)

# fileinfo = open('V2.0multi_data_len_pred.csv','w')
fileinfo = open('V2.0multi_data_theta_pred.csv','w')


for line1,line2,line3,line4 in zip(reader1,reader2,reader3,reader4):

	print line1[0]+','+line1[1]+'\n'+line2[0]+','+line2[1]+'\n'+line3[0]+','+line3[1]+'\n'+line4[0]+','+line4[1]
	fileinfo.write(line1[0]+','+line1[1]+'\n'+line2[0]+','+line2[1]+'\n'+line3[0]+','+line3[1]+'\n'+line4[0]+','+line4[1]+'\n')



csvfile1.close()