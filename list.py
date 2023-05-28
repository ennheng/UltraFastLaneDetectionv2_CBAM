import os
import json
import numpy as np

dir_json = 'C:\Users\Jason\Desktop\dataset\gaokong\json'  # json存储的文件目录
dir_txt = 'C:\Users\Jason\Desktop\dataset\gaokong\list_linshi'  # txt存储目录
if not os.path.exists(dir_txt):
    os.makedirs(dir_txt)
list_json = os.listdir(dir_json)


def json2txt(path_json, path_txt):  # 可修改生成格式
    with open(path_json, 'r') as path_json:
        jsonx = json.load(path_json)
        with open(path_txt, 'w+') as ftxt:
            for shape in jsonx['shapes']:
                label = str(shape['label']) + ' '
                xy = np.array(shape['points'])
                strxy = ''

                for m, n in xy:
                    m = int(m)
                    n = int(n)
                    # print('m:',m)
                    # print('n：',n)
                    strxy += str(m) + ' ' + str(n) + ' '

                label = strxy
                ftxt.writelines(label + "\n")


for cnt, json_name in enumerate(list_json):
    print('cnt=%d,name=%s' % (cnt, json_name))
    path_json = dir_json + json_name
    print(path_json)
    path_txt = dir_txt + json_name.replace('.json', '.lines.txt')
    print(path_txt)
    json2txt(path_json, path_txt)