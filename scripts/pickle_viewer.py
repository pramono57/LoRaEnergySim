import pickle
from pprint import pprint

file_path = './results/True_True_0.01_cnst_num_bytes_itm.p'

with open(file_path, 'rb') as f:
    data = pickle.load(f)

print('Keys:')
print(list(data.keys()))

print('\nData:')
pprint(data)
