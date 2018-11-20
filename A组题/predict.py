import csv
import pandas as pd
#csv_file =csv.reader(open('predictdata.csv','r'))
data=pd.read_csv('predictdata.csv')
data.head()
#print(data.isnull().sum())
print(data)
#import csv

##csv 写入
#stu1 = ['marry',26]
#stu2 = ['bob',23]
##打开文件，追加a
#out = open('Stu_csv.csv','a', newline='')
##设定写入模式
#csv_write = csv.writer(out,dialect='excel')
##写入具体内容
#csv_write.writerow(stu1)
#csv_write.writerow(stu2)
#print ("write over")
##缺失值删除处理
##https://blog.csdn.net/sinat_29957455/article/details/79418041