import sys
sys.path.append('/home/v-wanzho/v-wanzho/LSAT/code')
import json
import os
from rule_pattern import RulePattern
sys.path.append('/home/v-wanzho/v-wanzho/LSAT/code/data_analysis')
from program_search.API import non_triggers
# RP = RulePattern()
no_use = ['the','a']
def extract_program(info):
    return ''


'''
def parse_func_cp_tree(tree,sentence,words,tags):
    func2pos = {}
    for k,v in non_triggers.items():
        if isinstance(v[0],list):
            flags = []
            tmp_pos = []
            for v_sub in v:
                for trigger in v_sub:
                    if trigger in ['RBR','RBS','JJR','JJS']:
                        pos = [x_idx for x_idx,x in enumerate(tags) if trigger==x]
                        if pos:
                            flag = True
                            tmp_pos.append()
                            break

                    else:
                        pos = [x_idx for x_idx, x in enumerate(words) if trigger in x.lower()]
                        if ' '+trigger+' ' in ' '+sentence+' ':
                            flag = True
                            break
                    flags.append(flag)
            if all(flags):
                if k not in func2pos:
                    func2pos[k] = pos

        else:
            flag = False
            for trigger in v:

'''
def same(x1,x2):
    x1 = x1.lower()
    x2 = x2.lower()
    if x1 in no_use or x2 in no_use:
        return False
    return ((x1==x2) or (x1 in x2 and len(x1)>=2) or (x2 in x1 and len(x2)>=2))


def find_wspan(text,tokens,cspan):
    start = None
    end = None
    cand = []
    for i in range(len(tokens)):
        if tokens[i] in text:
            if start:
                end+=1
            else:
                start,end = i,i
        else:
            if start:
                start_dis = abs(cspan[0]-len(' '.join(tokens[:start])))
                end_dis = abs(cspan[1]-len(' '.join(tokens[:end+1])))
                cand.append((start,end,(start_dis+end_dis)/2))
                start,end=None,None
    if start:
        start_dis = abs(cspan[0] - len(' '.join(tokens[:start])))
        end_dis = abs(cspan[1] - len(' '.join(tokens[:end + 1])))
        cand.append((start, end, (start_dis + end_dis) / 2))

    cand = sorted(cand,key=lambda x:x[2],reverse=False)
    return (cand[0][0],cand[0][1]+1)

def match_words_tags_func(sentence,words,tags):
    sentence = sentence.replace('—',' ')
    sentence = sentence.replace('.','')
    func2pos = {}
    for k,v in non_triggers.items():
        if isinstance(v[0], list):
            flags = []
            tmp_pos = []
            for v_sub in v:
                flag=False

                for trigger in v_sub:
                    if trigger in ['RBR', 'RBS', 'JJR', 'JJS']:
                        pos = [x_idx for x_idx, x in enumerate(tags) if trigger == x]
                        if pos:
                            flag = True
                            tmp_pos.extend(pos)

                    else:
                        if ' ' + trigger + ' ' in ' ' + sentence + ' ':
                            pos = [x_idx for x_idx, x in enumerate(words) if same(x,trigger)]
                            flag = True
                            tmp_pos.extend(pos)
                flags.append(flag)
            if all(flags):
                if k not in func2pos:
                    func2pos[k] = tmp_pos
        else:
            tmp_pos = []
            for trigger in v:
                if trigger in ['RBR', 'RBS', 'JJR', 'JJS']:
                    pos = [x_idx for x_idx, x in enumerate(tags) if trigger == x]
                    tmp_pos.extend(pos)
                else:
                    if ' ' + trigger + ' ' in ' ' + sentence.lower() + ' ':
                        pos = [x_idx for x_idx, x in enumerate(words) if same(x, trigger)]
                        tmp_pos.extend(pos)

            if tmp_pos:
                func2pos[k] = tmp_pos

    pos2func = [[] for _ in range(len(words))]
    for k,v in func2pos.items():
        for v_sub in v:
            pos2func[v_sub].append(k)
    return func2pos,pos2func


def map_charid_to_token_id(text,tokens):
    start_idx = 0
    token2charid = {}
    charid2token = {}
    for tidx,token in enumerate(tokens):
        pos = text.find(token,start_idx)
        if pos!=-1:
            token2charid[tidx] = list(range(pos,pos+len(token)))
        start_idx = pos+len(token)
    for items in token2charid.items():
        for v in items[1]:
            charid2token[v] = items[0]
    # for cidx in range(len(text)):
    #     if cidx not in charid2token.keys():
    #         tmp_idx = cidx-1
    #         while tmp_idx not in charid2token.keys():
    #             tmp_idx-=1
    #         charid2token[cidx] = charid2token[tmp_idx]

    return token2charid,charid2token


import re
basic_dir = '../data/'
datap =  os.path.join(basic_dir, 'cp_ner_dp_results/AR_DevelopmentData_cp_ner_dp_results.json')
data = json.load(open(datap,'r',encoding='utf8'))
cond_datap = os.path.join(basic_dir, 'analyze_data/AR_DevelopmentData_analyze_condition.json')
cond_data = json.load(open(cond_datap,'r',encoding='utf8'))
for pidx,passage in enumerate(data):
    for sidx,sen_cp in enumerate(passage['sentence_cp_results']):
        func2pos, pos2func = match_words_tags_func(passage['sentences'][sidx],sen_cp['tokens'],sen_cp['pos_tags'])
        # print(sen_cp['tokens'],'\n')
        # print(sen_cp['pos_tags'],'\n')
        participants,columns = cond_data[pidx]['conditions']['participants'], cond_data[pidx]['conditions']['columns']
        results = [(sen_cp['tokens'][i],sen_cp['pos_tags'][i],pos2func[i]) for i in range(len(sen_cp['tokens']))]
        print(passage['sentences'][sidx])
        print(results,'\n')
        print(participants,columns)
        t2c,c2t = map_charid_to_token_id(passage['sentences'][sidx],[wd[0] for wd in results])
        # print(c2t)
        func_name = {}
        for widx,res in enumerate(results):
            if res[2]:
                names = res[2]
                for name in names:
                    pos = t2c[widx]
                    if name not in func_name.keys():
                        func_name[name] = [[widx]]
                    else:
                        func_pos = func_name[name]
                        flag = True
                        for idx,p in enumerate(func_pos):
                            dis = abs(max(p)-widx)
                            if dis == 1:
                                func_name[name][idx].append(widx)
                                flag=False
                        if flag:
                            func_name[name].append([widx])

        par_occur = []
        for participant in participants:
            par_results = re.finditer(participant, passage['sentences'][sidx])
            if par_results:
                for res in par_results:
                    tmp = {'participant': res.group(), 'position': (c2t[min(res.span())], c2t[max(res.span()) - 1])}
                    par_occur.append(tmp)

        col_occur = []
        for col in columns:
            col_results = re.finditer(col, passage['sentences'][sidx])
            if col_results:
                for res in col_results:
                    tmp = {'participant': res.group(), 'position': (c2t[min(res.span())], c2t[max(res.span()) - 1])}
                    col_occur.append(tmp)
        print(par_occur)
        print(func_name)
        input()