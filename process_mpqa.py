import os
import codecs
import json
import random
doc_dir = "database.mpqa.2.0/docs/"
ann_dir = "database.mpqa.2.0/man_anns/"
anno_file_name = "/gateman.mpqa.lre.2.0"
sent_file_name = "/gatesentences.mpqa.2.0"


def fetch_doc_content(doc_dir):
    doc_dir_list = os.listdir(doc_dir)
    doc_dir_list.remove('non_fbis')
    doc_dir_list.remove('temp_fbis')
    doc_dir_list.remove('ula')
    doc_dir_list.remove('xbank')
    fnames = {}
    for dir in doc_dir_list:
        flists = os.listdir(doc_dir + dir)
        for fname in flists:
            inFp = open(doc_dir + dir + '/' + fname, 'rb')
            doc_cont = inFp.read()
            inFp.close()
            fnames[dir + '/' + fname] = doc_cont
    return fnames

def fetch_sent_ids(flist):
    file_sent_id = {}
    for fp in flist:
        sent_id = []
        with codecs.open(ann_dir + fp + sent_file_name, 'r', 'utf-8') as inFp:
            data = inFp.readlines()
            inFp.close()
        data = [d.split('\t')[1] for d in data]
        data = [d.split(',') for d in data]
        for i in data:
            sent_id.append(list(map(int, i)))
        file_sent_id[fp] = sent_id
    return file_sent_id

def extract_info(file_list, file_context, file_sent_ids):
    data_info = {}
    for fp in file_list:
        data_info[fp] = {}
        data_info[fp]['extract'] = []
        data_info[fp]['context'] = str(file_context[fp])[2:-1]

        with codecs.open(ann_dir + fp + anno_file_name, 'r', 'utf-8') as inFp:
            data = inFp.readlines()
            inFp.close()

        valid_att = ['GATE_attitude', 'GATE_target', 'GATE_direct-subjective', 'GATE_agent']
        data = [d for d in data if d[0] != '#']
        data = [d.split('\t') for d in data]
        data = [[d[1], d[3], d[4][:-1].strip(' ')] for d in data]

        pre_data = []
        for t in data:
            if t[1] in valid_att:
                pt = t[2]
                pt = ','.join(pt.split(', '))
                t[2] = pt
                pre_data.append(t)

        ds_dict = {}
        at_dict = {}
        tt_dict = {}
        ss_dict = {}
        for t in pre_data:
            if t[1] == 'GATE_agent':
                if 'id' not in t[2]:
                    continue
                else:
                    seid = list(map(int, t[0].split(',')))
                    temp = t[2].split(' ')[0]
                    temp = temp.split('=')[1][1:-1]

                    if temp not in ss_dict:
                        ss_dict[temp] = file_context[fp][seid[0]:seid[1]]

            elif t[1] == 'GATE_direct-subjective':

                ds_dict[t[0]] = {}
                ds_dict[t[0]]['att'] = ''
                ds_dict[t[0]]['source'] = []
                anno_detail = t[2].split(' ')
                for idj, j in enumerate(anno_detail):
                    if 'attitude-link' in j:
                        temp = j.split('=')[1][1:-1]
                        if ',' in temp:
                            temp = temp.split(',')[0]
                        ds_dict[t[0]]['att'] = '"' + temp + '"'

                    elif 'nested-source' in j:
                        #print(j)
                        temp = j.split('=')[1][1:-1]
                        if ',' in temp:
                            temp = temp.split(',')
                            for a_t in temp:
                                ds_dict[t[0]]['source'].append(a_t)
                        else:
                            ds_dict[t[0]]['source'].append(temp)
                    else:
                        continue
            elif t[1] == 'GATE_attitude':
                anno_detail = t[2].split(' ')
                tlink = ''
                for j in anno_detail:
                    if 'target-link' in j:
                        tlink = j.split('=')[1][1:-1]
                        if ',' in tlink:
                            tlink = tlink.split(',')[0]
                        tlink = '"' + tlink + '"'
                    elif 'id' in j:
                        aid = j.split('=')[1]
                    else:
                        continue
                at_dict[aid] = tlink
            elif t[1] == 'GATE_target':
                # print(t)
                anno_detail = t[2]
                if ' ' in anno_detail:
                    anno_detail = anno_detail.split(' ')[0]

                tid = ''
                tspan = t[0].split(',')
                tspan = list(map(int, tspan))

                tid = anno_detail.split('=')

                tid = tid[1][1:-1]

                if ',' in tid:
                    temp = tid.split(',')[0]
                else:
                    temp = tid

                tt_dict['"' + temp + '"'] = tspan

        for dskey in ds_dict.keys():
            temp_dict = {}
            temp_dict['text'] = ""
            temp_dict['question'] = ""
            temp_dict['answer'] = []

            skip_flg = False

            if ds_dict[dskey]['att'] in at_dict:
                if at_dict[ds_dict[dskey]['att']] in tt_dict:
                    sid, eid = tt_dict[at_dict[ds_dict[dskey]['att']]]
                    sent_id = file_sent_ids[fp]
                    for m, n in sent_id:
                        if sid >= m and eid <= n:
                            temp_dict['text'] = str(file_context[fp][m:n])[2:-1]
                            break
                    temp_dict['question'] = str(file_context[fp][sid:eid])[2:-1]


                    for ans in ds_dict[dskey]['source']:
                        if ans == 'w' or ans == 'implicit':
                            if 'w' not in temp_dict['answer']:
                                temp_dict['answer'].append('w')
                        else:
                            if ans in ss_dict:
                                temp_dict['answer'].append(str(ss_dict[ans])[2:-1])

            for k in temp_dict.keys():
                if temp_dict[k] == '':
                    skip_flg = True
                    break

            #for k in temp_dict['answer']:
            #    if k.lower() not in str(file_context[fp]).lower():
            #        print(k, str(file_context[fp]).lower().split('\\'))
            #        skip_flg = True
            # exit()
            if skip_flg == False:
                data_info[fp]['extract'].append(temp_dict)

        if len(data_info[fp]['extract']) == 0:
            del data_info[fp]
    return data_info

file_context = fetch_doc_content(doc_dir)
file_list = list(file_context.keys())
file_sent_ids = fetch_sent_ids(file_list)
data = extract_info(file_list, file_context, file_sent_ids)

total_ins = len(data.keys())
train_num = int(0.9 * total_ins)
test_num = int(0.1 * total_ins)
total_key = list(data.keys())

path = "cross_validation/"
for iter in range(10):
    os.mkdir(path + str(iter))
    new_path = path + str(iter)

    random.seed(iter)
    random.shuffle(total_key)

    data_train = total_key[:train_num]
    data_test = total_key[train_num:]

    json_train = {}
    json_test = {}
    for i in data_train:
        json_train[i] = data[i]

    for i in data_test:
        json_test[i] = data[i]

    with open(new_path + '/mpqa_extract_data_nested_train.json', 'w') as outfile:
        json.dump(json_train, outfile)
        outfile.close()

    with open(new_path + '/mpqa_extract_data_nested_test.json', 'w') as outfile:
        json.dump(json_test, outfile)
        outfile.close()