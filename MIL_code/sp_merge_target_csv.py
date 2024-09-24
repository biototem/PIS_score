import pandas
from my_py_lib.path_tool import *

big_csv = {}

for file in find_file_by_exts('/mnt/totem_new/projectM_data/slide_data/tron_data_cls_xlsx', '.xlsx'):
    table = pandas.read_excel(file)
    for i in range(table.shape[0]):
        name = table.iloc[i][0]
        idx = int(name[2:])
        value = table.iloc[i][1].replace('\r', '').replace('\n', '')
        big_csv[idx] = [name, value]

# print(big_csv)

table = pandas.read_excel('/mnt/totem_data/totem/fengwentai/download/organized.xlsx', 'test')

new_col1 = []
new_col2 = []

def tr_relpath_to_id(n: str):
    n = n.split('/')[-1]
    ns = n.split('.')
    i_start = int(ns[0].split('-')[0])
    i_add = int(ns[2].split('-')[-1])
    i = i_start + i_add
    return str(i)

for relpath in table[' relpath'].tolist():
    if 'kfb_data' in relpath:
        new_col1.append('')
        new_col2.append('')
    else:
        idx = int(tr_relpath_to_id(relpath))
        if idx in big_csv:
            new_col1.append(big_csv[idx][0])
            new_col2.append(big_csv[idx][1])
        else:
            new_col1.append('')
            new_col2.append('')

table['检查号'] = new_col1
table['检查结果'] = new_col2

table.to_excel('sp_out2.xlsx')
