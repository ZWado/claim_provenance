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

data2 = random.sample(data2, len(data1))
print(len(data2))


with open(id + '/train_pairwise_positive_srl_sample.json', 'w') as outfile:
    json.dump(data2, outfile)
    outfile.close()