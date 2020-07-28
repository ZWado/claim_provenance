import json
import random

id = '9'

with open(id + '/train_new_pairwise.json', 'r') as read_file:
    data1 = json.load(read_file)
    read_file.close()

print(len(data1))

with open(id + '/train_pairwise_srl_replaced.json', 'r') as read_file:
    data2 = json.load(read_file)
    read_file.close()

print(len(data2))

data = data1 + data2
print(len(data))

with open(id + '/train_pairwise_positive_srl_replaced_full.json', 'w') as outfile:
    json.dump(data, outfile)
    outfile.close()