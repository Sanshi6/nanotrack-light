
# import json
# import os, sys
#
# json_path = r'E:\SiamProject\NanoTrack\datasets\VOT2018\VOT2018.json'
# new_json_path = r'E:\SiamProject\NanoTrack\datasets\VOT2018\VOT2018_new.json'
# txt_path = r'E:\SiamProject\NanoTrack\datasets\VOT2018\list.txt'
# dict = {}
#
#
# def get_json_data(path):  # 修改 删除原json文件中的color
#     with open(path) as f:
#         params = json.load(f)
#         file = open(txt_path)
#         while 1:
#             lines = file.readline(1000)
#             if not lines:
#                 break
#             lines = lines[:-1]  # 拿出每个视频文件夹的名字
#             root = (params[lines]['img_names'])
#             for i in range(len(root)):
#                 kind, color, jpg = root[i].split('/')  # 举例 kind :'ants1', color: 'color' , jpg :'00000001.jpg'
#                 root[i] = kind + '/' + jpg  # 重写该路径，去掉 color
#         file.close()
#         dict = params
#         # print(dict)
#     f.close()
#     return dict
#
#
# def write_json_data(path, dictionary):  # 保存
#     with open(path, 'w') as r:
#         json.dump(dictionary, r)
#     r.close()
#     print("done.")
#
#
# if __name__ == '__main__':
#     dictionary = get_json_data(json_path)
#     write_json_data(new_json_path, dictionary)





from openpyxl import Workbook
from datetime import datetime, timedelta

# 新建 workbook 和 sheet
wb = Workbook()
ws = wb.active

# 定义开始日期
start_date = datetime(2023, 9, 4)

# 用循环来填充日期
for i in range(20):  # 填充10周的日期
    end_date = start_date + timedelta(days=6)
    date_range = "{}-{}".format(start_date.strftime("%Y.%m.%d"), end_date.strftime("%Y.%m.%d"))
    ws.append([date_range])
    # 设置下一个日期范围的开始日期
    start_date = end_date + timedelta(days=1)

# 保存
wb.save("date_ranges.xlsx")