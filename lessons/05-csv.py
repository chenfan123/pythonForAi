from python_for_ai import get_llm_response, print_llm_response
import csv

f = open('lessons/05-file.csv', 'r') # 以读取模式打开文件

csv_read = csv.DictReader(f) # 告诉python这是一个csv，把每一行转换成字典的工具
itinerary = []
for row in csv_read:
    print(row) # {'列1': 'a', '列2': 'a-1', '列3': 'a-2', '列4': 'a-3'}
    itinerary.append(row)
print(itinerary)
f.close()