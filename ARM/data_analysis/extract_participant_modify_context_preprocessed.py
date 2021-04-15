import json
import os
import nltk
import re
from tqdm import tqdm
from copy import deepcopy
import sys
import copy
from rule_pattern import RulePattern
from collections import Counter
RP = RulePattern()
def rank_entities(entities):
    quantity_ents = []
    rest_ents = []
    flag = False
    for group in [RP.simple_order,RP.orders, RP.numbers, RP.week_words, RP.month_words]:
        if any([ent in group for ent in entities]) or any([ent.isnumeric() for ent in entities]):
            for ent in entities:
                if ent in group.keys():
                    quantity_ents.append((ent, group[ent]))
                elif ent.isnumeric():
                    quantity_ents.append((ent, int(ent)))
                else:
                    rest_ents.append((ent, None))
            flag = True
            break
    ranked_quan_ent = sorted(quantity_ents, key=lambda k: k[1])
    all_entities = [ent[0] for ent in ranked_quan_ent + rest_ents] if flag else entities
    return all_entities

basic_dir = '../../data'
datap = os.path.join(basic_dir, 'AR_TestData_cp_ner_dp_results.json')#'cp_ner_dp_results/AR_DevelopmentData_cp_ner_dp_results.json')
outp =  os.path.join(basic_dir, 'new_data/ar_test_analyze_condition.json')#'model_data/ar_val_analyze_condition.json')
passages = json.load(open(datap, 'r',encoding='utf8'))

all_passages = []
all_output_for_model = []
ans2label = {'A':0,'B':1,'C':2,'D':3,'E':4}
for pidx,passage in enumerate(passages):
    doc = passage['passage']#incase the first word is a number

    ques = passage['questions']
    sens = [sen for sen in passage['sentences']]


    conditions = {'participant_group':[],'position_group':[]}
    # print('--------------------------------------------------')
    # print(sens)
    sen_tokens, sen_ent_res = passage['sentence_cp_results'],passage['sentence_ner_results']
    sen_num = len(sens)
    all_kws = []
    all_sen_kws = []
    #extract conditions
    doc_ent_group,doc_num2ent,all_tokens = [],[],[]
    start_idx = 0
    passages[pidx]['sentence_keywords']=[]
    for idx in range(len(sens)):
        sen, tokens,ent_res = sens[idx],sen_tokens[idx],sen_ent_res[idx]
        re_pattern_sen_kws = RP.found_key_words_leading_sen(sen, tokens, ent_res,start_idx)
        if idx<3:
            all_tokens.extend(sen_tokens[idx]['tokens'])
            entity_group = re_pattern_sen_kws['entity_group']

            quantity2group = re_pattern_sen_kws['range_quantity_pairs']

            if entity_group:
                doc_ent_group.extend(entity_group.copy())

            doc_num2ent.extend(quantity2group)
            start_idx = len(all_tokens)

        sen_kws = RP.found_key_words_common(sen,tokens,ent_res)
        sen_kws['entity_group'] = copy.deepcopy(re_pattern_sen_kws['entity_group'])
        sen_kws['quantity_pairs'] = copy.deepcopy(re_pattern_sen_kws['quantity_pairs'])

        all_sen_kws.append(sen_kws)
        passages[pidx]['sentence_keywords'].append(sen_kws)
        cond_ent = RP.combine_cond_keywords(sen_kws)
        for group in cond_ent['entity_group']:
            all_kws.extend(group['entities'])
    # print(doc_ent_group)
    # input()
    participants, columns,modified_doc = RP.extract_participants_columns(doc_ent_group, doc_num2ent, all_tokens,doc)
    columns = rank_entities(list(columns))
    participants = rank_entities(participants)
    print(doc)
    if participants == columns:
        # print(participants)
        try:
            columns = [list(RP.orders.keys())[idx] for idx in range(len(participants))]
        except Exception as e:
            print(e)

    participants = [tmp for tmp in participants if tmp]
    columns = [tmp for tmp in columns if tmp]
    print(participants, columns)
    print('------------------')
    # participants, columns = RP.extract_participants_columns(doc_ent_group, doc_num2ent, all_tokens)
    #analyze keywords in question and answer
    ques_text = [que['question'] for que in ques]

    all_ques_kws = []
    all_opt_kws = []
    for idx in range(len(ques)):
        # que_kws = RP.extract_question_keywords(ques[idx]['question'],ques_tokens[idx],ques_ent_res[idx])
        que_kws = RP.found_key_words_common(ques[idx]['question'],ques[idx]['question_cp_results'],ques[idx]['question_ner_results'])
        clean_que_kws = RP.combine_common_keywords(que_kws)
        all_ques_kws.append(que_kws)
        passages[pidx]['questions'][idx]['question_keywords'] = clean_que_kws
        all_kws.extend(clean_que_kws['filtered_keywords'])
        # print(ques[idx]['question'])
        # print(clean_que_kws)
        options = ques[idx]['options']
        # print(options)
        opt_tokens, opt_ent_res = ques[idx]['option_cp_results'],ques[idx]['option_ner_results']#RP.get_cp_ner_results(options)
        opt_kws = []
        passages[pidx]['questions'][idx]['options_keywords'] = []
        #output for model
        # if doc!=modified_doc:
        #     print(doc,'\n',modified_doc,'\n','---------------------')

        # modified_options = RP.modify_option(options)
        # print(options,'\n',modified_options)

        all_output_for_model.append({'context':doc,'question':ques[idx]['question'],'answers':options,'label':ans2label[ques[idx]['answer']],
                                     'rows':list(participants),'columns':list(columns),'id_string':ques[idx]['id'],'tags':ques[idx]['tags']})
        for opt_idx in range(len(options)):
            # tmp_kws = RP.extract_option_keywords(options[opt_idx],opt_tokens[opt_idx],opt_ent_res[opt_idx])
            tmp_kws = RP.found_key_words_common(options[opt_idx],opt_tokens[opt_idx],opt_ent_res[opt_idx])
            clean_tmp_kws = RP.combine_common_keywords(tmp_kws)
            opt_kws.append(clean_tmp_kws)
            all_opt_kws.append(opt_kws)
            passages[pidx]['questions'][idx]['options_keywords'].append(opt_kws)
            all_kws.extend(clean_tmp_kws['filtered_keywords'])
        # print('*********************')
    #calculate the frequency of mentioned entities to see which are important
    count = Counter(all_kws)
    #information summarization
    passages[pidx]['conditions'] = {}
    passages[pidx]['conditions']['participants'] = participants
    passages[pidx]['conditions']['columns'] = columns
    passages[pidx]['entity_frequency'] = count
    passages[pidx]['rules'] = sens



with open(outp,'w',encoding='utf8') as outf:
    json.dump(all_output_for_model,outf)

with open(outp.replace('.json','beautiful.json'),'w',encoding='utf8') as outf:
    json.dump(all_output_for_model,outf,indent=4)





