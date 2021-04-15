import json
import os
# nltk.download('punkt')
import re

import nltk

from API import API

API = API()
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
import copy
import sys
from answer_question_by_tree import QuestionType,choose_question_type
from rule_pattern import RulePattern
from rule_update_table import *#RuleTree,Node,merge_funcs_with_or,merge_funcs_with_iff,merge_option_functions
RP = RulePattern()
# list(number_of_exams) -> dict ('sections') (list of sections,select 0) -> dict(passages)
# -> list (number of passages) -> dict(['id',
# 'questions', 'fatherId', 'passageId', 'passage', 'passageWithLabel'])

orders = {
    'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'fifth': 5, 'sixth': 6, 'seventh': 7,
    'eighth': 8, 'ninth': 9, 'tenth': 10, 'eleventh': 11, 'twelfth': 12, 'thirteenth': 13,
    'fourteenth': 14, 'fifteenth': 15, 'sixteenth': 16
}
week_words = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7}
month_words = {
    "January": 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6, 'July': 7, 'August': 8, 'September': 9,
    'October': 10,
    'November': 11, 'December': 12
}
negation_words = [wd + ' ' for wd in ['not', 'nor', 'no', 'never', "didn't", "won't", "wasn't", "isn't",
                                      "haven't", "weren't", "won't", 'neither', 'none', 'unable',
                                      'fail', 'different', 'outside', 'unable', 'fail', 'cannot', 'except', 'CANNOT']]

all_num_words = []
for dic in RP.all_num_dict:
    all_num_words.extend(list(dic.keys()))
def clean_doc(doc):
    doc = ' ' + doc
    doc = doc.replace('—', ' — ')
    doc = doc.replace('—', ' — ')
    doc = doc.replace('\u2014', ' ')
    return doc


def extract_facts_rules(sentences, rows, columns):
    # sentences = nltk.sent_tokenize(context)
    all_fact_flag = [False] * len(sentences)
    rules, facts = [], []
    tmp_rows = rows
    tmp_columns = [] if rows == columns else columns
    par_counts,col_counts = [0],[0]
    for idx, sen in enumerate(sentences[1:]):
        par_exist = [tmp for tmp in tmp_rows if tmp in sen]
        col_exist = [tmp for tmp in tmp_columns if tmp in sen]
        par_count = len(par_exist)
        par_counts.append(par_count)
        col_count = len(col_exist)
        col_counts.append(col_count)
        # print('exist situation: ',par_exist,col_exist)
        if par_count >= 1 and col_count >= 1:
            all_fact_flag[idx + 1] = True
        elif (par_count >= 2 and col_count == 0) or (col_count >= 2 and par_count == 0):
            all_fact_flag[idx + 1] = False
        # else:
        fact_kws = ['if ', 'If ', 'Then ', 'then ']
        for kw in fact_kws:
            if kw in sen:
                all_fact_flag[idx + 1] = False
                break

        words = nltk.word_tokenize(sen)
        tags = nltk.pos_tag(words)
        func2pos, pos2func = match_words_tags_func(sen, words, tags,
                                                   limit=[k for k in API.non_triggers.keys() if k != 'to'])
        if func2pos.keys():
            all_fact_flag[idx+1] = False

    for idx, flag in enumerate(all_fact_flag):
        if (par_counts[idx]>=2/3*len(rows) or col_counts[idx]>=2/3*len(columns)) and len(columns)>2:
            continue
        if idx == 0:  # skip the first sentence
            continue
        if flag:
            facts.append(idx)
        else:
            rules.append(idx)
    return facts, rules

def extract_if_then(sentence):
    iff_kws = ['if and only if','if but only if']
    if_kws = ['If ','if ']
    then_kws = ['then','Then','so does','which','when','who',',','.']
    flag = False
    if_pos,then_pos = 0,None
    iff_flag = False
    for kw in iff_kws:
        tmp_sentence = sentence.replace(',','')
        if kw in tmp_sentence:
            start_pos = tmp_sentence.find(kw)
            end_pos = start_pos+len(kw) if start_pos!=-1 else 0
            iff_flag = True
            return iff_flag,flag,start_pos,end_pos,tmp_sentence
    for kw in if_kws:
        if kw in sentence:
            if_pos = sentence.find(kw)
            for tkw in then_kws:
                if tkw in sentence[if_pos+len(kw):]:
                    then_pos = sentence.find(tkw,if_pos+len(kw))
                    break
            flag = True
            break
    if not then_pos:
        then_pos = len(sentence)
    # print('instance analysis: ',sentence,flag,sentence[if_pos:then_pos], sentence[then_pos:])
    return iff_flag,flag,if_pos,then_pos,sentence #sentence[if_pos:then_pos], sentence[then_pos:],if_pos,then_pos




def same(x1, x2):
    no_use = ['the', 'a']
    x1 = x1.lower()
    x2 = x2.lower()
    if x1 in no_use or x2 in no_use:
        return False
    return (x1 == x2)  # or (x1 in x2 and len(x1)>=2) or (x2 in x1 and len(x2)>=2))


def find_wspan(text, tokens, cspan):
    start = None
    end = None
    cand = []
    for i in range(len(tokens)):
        if tokens[i] in text:
            if start:
                end += 1
            else:
                start, end = i, i
        else:
            if start:
                start_dis = abs(cspan[0] - len(' '.join(tokens[:start])))
                end_dis = abs(cspan[1] - len(' '.join(tokens[:end + 1])))
                cand.append((start, end, (start_dis + end_dis) / 2))
                start, end = None, None
    if start:
        start_dis = abs(cspan[0] - len(' '.join(tokens[:start])))
        end_dis = abs(cspan[1] - len(' '.join(tokens[:end + 1])))
        cand.append((start, end, (start_dis + end_dis) / 2))

    cand = sorted(cand, key=lambda x: x[2], reverse=False)
    return (cand[0][0], cand[0][1] + 1)


def match_words_tags_func(sentence, words, tags, limit=None):
    # sentence = sentence.replace('—',' ')
    # sentence = sentence.replace('.','')
    func2pos = {}
    for k, vv in API.non_triggers.items():
        if limit:
            if k not in limit:
                continue
        for v in vv:
            if any([isinstance(tmp, list) for tmp in v]):
                flags = []
                tmp_pos = []
                for v_sub in v:
                    flag = False
                    for trigger in v_sub:
                        if trigger in ['RBR', 'RBS', 'JJR', 'JJS']:
                            pos = [x_idx for x_idx, x in enumerate(tags) if trigger == x]
                            if pos:
                                flag = True
                                tmp_pos.extend(pos)

                        else:
                            if ' ' + trigger + ' ' in ' ' + sentence + ' ':
                                pos = [x_idx for x_idx, x in enumerate(words) if same(x, trigger)]
                                flag = True
                                tmp_pos.extend(pos)
                    flags.append(flag)
                if all(flags):
                    tmp_pos = sorted(tmp_pos)
                    pos_pair = []
                    for id,p in enumerate(tmp_pos):
                        if id == 0:
                            pos_pair.append([p])
                        else:
                            if p - pos_pair[-1][-1] == 1:
                                pos_pair[-1].append(p)
                            else:
                                pos_pair.append([p])
                    for id,pair in enumerate(pos_pair):
                        if len(pair)<len(v):
                            pos_pair.pop(id)

                    if k not in func2pos:
                        func2pos[k] = pos_pair
                    else:
                        func2pos[k].extend(pos_pair)
            else:
                tmp_pos = []
                for trigger in v:
                    if trigger in ['RBR', 'RBS', 'JJR', 'JJS']:
                        pos = [x_idx for x_idx, x in enumerate(tags) if trigger == x]
                        tmp_pos.append(pos)
                    else:
                        if ' ' + trigger + ' ' in ' ' + sentence.lower() + ' ':
                            pos = [x_idx for x_idx, x in enumerate(words) if same(x, trigger)]
                            tmp_pos.append(pos)

                if tmp_pos:
                    func2pos[k] = tmp_pos

    pos2func = [[] for _ in range(len(words))]
    # print(func2pos.items())
    for k, vv in func2pos.items():
        for v in vv:
            for v_sub in v:
                if len(pos2func[v_sub])==0:
                    pos2func[v_sub].append(k)
    # for item in pos2func:
    #     if len(item)>1:

    return func2pos, pos2func


def map_charid_to_token_id(text, tokens):
    # print(text,tokens)
    start_idx = 0
    token2charid = {}
    charid2token = {}
    for tidx, token in enumerate(tokens):
        pos = text.find(token, start_idx)
        if pos != -1:
            token2charid[tidx] = list(range(pos, pos + len(token)))
            start_idx = pos + len(token)
    for items in token2charid.items():
        for v in items[1]:
            charid2token[v] = items[0]
    for cidx in range(len(text)):
        if cidx not in charid2token.keys():
            if cidx!=0:
                charid2token[cidx] = charid2token[cidx - 1]
            else:
                charid2token[cidx] = 0

    return token2charid, charid2token


def analyze_fact(fact, rows, columns, table):
    # print(fact)
    parts = fact.split(';')
    for part in parts:
        par_exist = [idx for idx, tmp in enumerate(rows) if tmp in part]
        col_exist = [idx for idx, tmp in enumerate(columns) if tmp in part]
        if any([wd in part for wd in negation_words]):
            v = False
        else:
            v = True
        for par in par_exist:
            for col in col_exist:
                # print(rows[par],columns[col],v)
                table[par][col] = v
    return table

def new_analyze_fact(fact,rows,columns,old_assignments):
    parts = fact.split(';')
    all_assignments = old_assignments
    can_found_exist = [False for i in range(len(parts))]

    for p,part in enumerate(parts):
        par_exist = []#[tmp for idx, tmp in enumerate(rows) if tmp in part]
        col_exist = []#[tmp for idx, tmp in enumerate(columns) if tmp in part]
        for idx,tmp in enumerate(rows):
            if tmp in part:
                par_exist.append((tmp,part.find(tmp)))
        for idx,tmp in enumerate(columns):
            if tmp in part:
                col_exist.append((tmp,part.find(tmp)))
        sorted_par_exist = sorted(par_exist,key=lambda k:k[1])
        and_pos = []
        if ' and ' in part:
            and_pos = [m.start() for m in re.finditer(' and ', part)]#part.find_all(' and ')#A in X and B in Y
        and_pos = [0]+and_pos+[len(part)] if and_pos else [0,len(part)]
        par_and_pos = []
        sorted_col_exist = sorted(col_exist, key=lambda k: k[1])
        for idx,par in enumerate(sorted_par_exist):
            for j,pos in enumerate(and_pos[1:]):
                if par[1] in range(and_pos[j],pos):
                    par_and_pos.append((and_pos[j],pos))
                    break
        assert(len(par_and_pos)==len(sorted_par_exist))

        if par_exist and col_exist:
            can_found_exist[p] = True
            if any([wd in part for wd in negation_words]):
                v = False
            else:
                v = True
            for pid,par in enumerate(sorted_par_exist):
                #if multiple column is split by or, should be split into multiple assignment
                should_split = [False for i in range(len(sorted_col_exist))]
                split_pair = []
                for i, col in enumerate(sorted_col_exist[1:]):
                    inner_part = part[sorted_col_exist[i][1]:col[1]+1]
                    if ' or ' in inner_part:
                        split_pair.append((i-1,i))
                        should_split[i] = should_split[i-1] = True
                old = []
                for i, flag in enumerate(should_split):
                    if flag==False and (sorted_col_exist[i][1] in range(par_and_pos[pid][0],par_and_pos[pid][1])):
                        old.append((par[0],sorted_col_exist[i][0],v))
                if all_assignments:
                    for i in range(len(all_assignments)):
                        all_assignments[i].extend(old)
                else:
                    all_assignments=[old]

                for pair in split_pair:
                    cache = []
                    if all_assignments:
                        for a in all_assignments:
                            cache.append(a+[(par[0],sorted_col_exist[pair[0]][0],v)])
                            cache.append(a+[(par[0],sorted_col_exist[pair[1]][0],v)])
                    else:
                        cache.append([(par[0],sorted_col_exist[pair[0]][0],v)])
                        cache.append([(par[0], sorted_col_exist[pair[1]][0], v)])
                    all_assignments = cache
    #find conflict
    conflict_flag = False
    # print(all_assignments)
    for i,assign in enumerate(all_assignments):
        for item in assign:
            neg_item = list(copy.deepcopy(item))
            neg_item[2] = False if item[2] else True
            if tuple(neg_item) in assign:
                all_assignments.pop(i)
                conflict_flag = True
                break
            if item[2]:
                for col in columns:
                    if (tuple([item[0],col,True]) in assign and item[1]!=col):
                        all_assignments.pop(i)
                        conflict_flag = True
                        break
            if conflict_flag:
                break

    return all_assignments,conflict_flag, any(can_found_exist)

def assign_table_with_facts(rows, columns, facts):
    # table = [[None for col in columns] for row in rows]
    old_assignments = []
    add_rules = []
    for fact in facts:
        old_assignments,conflict_flag,can_found_exist = new_analyze_fact(fact, rows, columns, old_assignments)
        if not can_found_exist:
            add_rules.append(fact)
        if conflict_flag and not old_assignments:
            return [],[]

        # print('The fact is: ',fact)
        # print(table)

    if rows == columns:
        for i,assign in enumerate(old_assignments):
            for row in rows:
                old_assignments[i].append((row,row,True))
    # print(table)
    return old_assignments,add_rules



def find_par_col_occur(text, char2token, tokens_func_res, participants, columns):
    results = tokens_func_res
    c2t = char2token
    func_name2pos = {}
    for widx, res in enumerate(results):
        if res[2]:
            names = res[2]
            for name in names:
                # pos = t2c[widx]
                if name not in func_name2pos.keys():
                    func_name2pos[name] = [[widx]]
                else:
                    func_pos = func_name2pos[name]
                    flag = True
                    for idx, p in enumerate(func_pos):
                        dis = abs(max(p) - widx)
                        if dis == 1:
                            func_name2pos[name][idx].append(widx)
                            flag = False
                    if flag:
                        func_name2pos[name].append([widx])

    par_occur = []
    for pidx, participant in enumerate(participants):
        par_results = re.finditer(participant, text)
        if par_results:
            for res in par_results:
                # print(c2t)
                if res.group():
                    tmp = {'participant': res.group(),
                           'position': (c2t[min(res.span())], c2t[max(res.span()) - 1]),
                           'type': 'row', 'idx': pidx}
                    par_occur.append(tmp)

    col_occur = []
    for cidx, col in enumerate(columns):
        col_results = re.finditer(col, text)
        if col_results:
            for res in col_results:
                tmp = {'participant': res.group(),
                       'position': (c2t[min(res.span())], c2t[max(res.span()) - 1]),
                       'type': 'column', 'idx': cidx}
                col_occur.append(tmp)
    return par_occur, col_occur, func_name2pos


def derive_functions_from_rules(rows_cp_results, rules, rows, columns, table):
    rule_functions = []
    for sidx, rule_cp in enumerate(rows_cp_results):
        functions = []
        func2pos, pos2func = match_words_tags_func(rules[sidx], rule_cp['tokens'], rule_cp['pos_tags'])
        # print(sen_cp['tokens'],'\n')
        # print(sen_cp['pos_tags'],'\n')
        participants, columns = rows, columns
        results = [(rule_cp['tokens'][i], rule_cp['pos_tags'][i], pos2func[i]) for i in
                   range(len(rule_cp['tokens']))]

        t2c, c2t = map_charid_to_token_id(rules[sidx], [wd[0] for wd in results])
        # print(c2t)

        par_occur, col_occur, func_name2pos = find_par_col_occur(rules[sidx], c2t, results, rows, columns)


        for func, positions in func_name2pos.items():
            for pos in positions:
                if par_occur or col_occur:
                    func_str = getattr(API, 'func_{}'.format(func))(par_occur, col_occur, rule_cp['tokens'],
                                                                    pos, table)
                    functions.append(func_str)

    rule_functions.extend(functions)
    return rule_functions


from modify_option import analyze_question


def select_ent(part, rows, columns):
    part = str(part)
    for ridx, row in enumerate(rows):
        if part in row or row in part:
            return 'row', ridx, row
    for cidx, col in enumerate(columns):
        if part in col or col in part:
            return 'col', cidx, col
    return None, None


def search_exist_ent(phrase, rows, columns):
    part = str(phrase)
    all_exist = []
    for ridx, row in enumerate(rows):
        if row in part:
            all_exist.append(('row', ridx, part.find(row), row))
    for cidx, col in enumerate(columns):
        if col in part:
            all_exist.append(('col', cidx, part.find(col), col))
    return all_exist


def modify_table_for_option(parts, mapping_ents, rows, columns, table, contain_negation=True):
    for pidx, part in enumerate(parts):
        ent_type, eidx = select_ent(part, rows, columns)
        map_ent = mapping_ents[pidx]
        map_ent_type, map_eidx = select_ent(map_ent, rows, columns)
        if (ent_type and map_ent_type) and ent_type != map_ent_type:
            assign_row = eidx if ent_type == 'row' else map_eidx
            assign_col = map_eidx if map_ent_type == 'col' else eidx
            # print(table)
            # print(rows,columns)
            # print(assign_row,assign_col,part,map_ent)
            table[assign_row][assign_col] = False if contain_negation else True

    return table


def match_qa_func_and_modify_table(question, option, rows, columns, table):
    functions = []
    related_triggers = ['to', 'next', 'same']
    qa_pair = question + ' ' + option
    # print(qa_pair)
    words = nltk.word_tokenize(qa_pair)
    neg_count = 0
    for wd in words:
        if wd in negation_words:
            neg_count += 1
    negation = True if neg_count % 2 == 1 else False
    tags = nltk.pos_tag(words)
    func2pos, pos2func = match_words_tags_func(qa_pair, words, tags)#, limit=related_triggers)
    results = [(words[i], tags[i], pos2func[i]) for i in range(len(words))]
    t2c, c2t = map_charid_to_token_id(qa_pair, [wd[0] for wd in results])
    par_occur, col_occur, func_name2pos = find_par_col_occur(qa_pair, c2t, results, rows, columns)

    for func, positions in func_name2pos.items():
        for pos in positions:
            if par_occur or col_occur:
                func_str = getattr(API, 'func_{}'.format(func))(par_occur, col_occur, words, pos,
                                                                              table, negation, rows,columns)

                functions.extend(func_str)
                # table, func_str = getattr(API, 'func_{}'.format(func))(par_occur, col_occur, words, pos,
                #                                                               table, negation)
    #find all the assignment under searched functions
    all_assignment = []
    # print(functions)
    for func in functions:
        tmp = []
        tmp_assign = func.find_option_assignment()
        if all_assignment:
            for i,cond in enumerate(tmp_assign):
                for j, pre in enumerate(all_assignment):
                    tmp.append(cond+pre)
        else:
            tmp = tmp_assign
        all_assignment = tmp
    # all_assignment = list(set(all_assignment))
    # print(all_assignment)

    return all_assignment,functions


def analyze_option_with_pattern(option, rows, columns, sorted_que_ents, possible_que_columns,question):
    possible_assignment = []
    backup_flag = False
    match_pattern_1 = '([A-Za-z0-9\' ]+: [$,A-Za-z0-9\' ]+; )+[:$,A-Za-z0-9\' ]+'  # Jackson: Y; Larabee: W; Paulson: X; Torillo: Z
    match_pattern_2 = '([A-Za-z0-9\' ]+: [$,A-Za-z0-9\' ]+ )+[:$,A-Za-z0-9\' ]+'  # Raimes: Frank Sicoli: Gina, Hiro, Kevin Thompson: Laurie
    match_pattern_3 = '([A-Za-z0-9\' ]+(,|;) ){2,}([a-zA-Z0-9\' ]+)'  # 2, 3, 4, 5 or 2;3;4;5
    res1 = re.search(match_pattern_1, option)
    if res1:
        match_content = res1.group()
        parts = match_content.split('; ')
        for part in parts:
            if part.split(': ')==2:
                subj, obj = part.split(': ')
                subj_ents, obj_ents = [], []
                for s in re.split('(,|(and))\s', subj):  # subj.split(','):
                    s_ent = select_ent(s.strip(), rows, columns)
                    if s_ent[0]:
                        subj_ents.append(s_ent)
                for o in obj.split(','):
                    o_ent = select_ent(o.strip(), rows, columns)
                    if o_ent[0]:
                        obj_ents.append(o_ent)
                # subj_ents = search_exist_ent(subj,rows,columns)
                # obj_ents = search_exist_ent(obj,rows,columns)
                for ent1 in subj_ents:
                    for ent2 in obj_ents:
                        possible_assignment.append((ent1, ent2))
        if possible_assignment:
            return possible_assignment,backup_flag

    res2 = re.search(match_pattern_2, option)
    if res2:  # Raimes: Frank Sicoli: Gina, Hiro, Kevin Thompson: Laurie
        match_content = res2.group()
        parts = match_content.split(': ')
        all_ents = []
        # print(parts)
        for part in parts:
            ents = search_exist_ent(part, rows, columns)
            all_ents.append(ents)
        # print(all_ents)
        if all_ents[0]:
            subjs = all_ents[0]
            for eidx, ents in enumerate(all_ents[1:]):
                if ents:
                    sorted_ents = sorted(ents, key=lambda k: k[2])
                    if eidx != len(all_ents) - 2:
                        objs = sorted_ents[:-1]
                    else:
                        objs = sorted_ents
                    for subj in subjs:
                        for obj in objs:
                            possible_assignment.append((subj, obj))
                    subjs = [sorted_ents[-1]]
                else:
                    subjs = []
        # print('possible assignment',possible_assignment)
        if possible_assignment:
            return possible_assignment,backup_flag

    res3 = re.finditer(match_pattern_3, option)
    order_flag = any([tmp in list(orders.keys()) for tmp in sorted_que_ents])
    if res3:
        for res in res3:

            # parts.extend(res.group().split(', '))
            parts = re.split('[,;]\s', res.group())  # res.group().split(', ')

            if len(parts) <= len(sorted_que_ents):
                objs = [tmp['span'] for tmp in sorted_que_ents]
            elif possible_que_columns and len(parts) <= len(possible_que_columns):
                objs = possible_que_columns
            elif len(sorted_que_ents) == 1 and order_flag == False:
                objs = [sorted_que_ents[0]['span'] for i in range(len(parts))]
            else:
                objs = list(orders.keys())

            for pidx, part in enumerate(parts):
                subsubj = part.split(' and ')
                for s in subsubj:
                    sent = select_ent(s, rows, columns)
                    obj = select_ent(objs[pidx], rows, columns)
                    possible_assignment.append((sent, obj))

        if not possible_assignment:
            row_ent_occur, col_ent_oocur = extract_par_col(option,rows,columns)
            for r in row_ent_occur:
                for c in col_ent_oocur:
                    possible_assignment.append((r['span'],c['span']))
                    backup_flag = True
        return possible_assignment,backup_flag


def modify_table_with_assignment(table, assignments, contain_negation):
    for assign in assignments:
        subj, obj = assign
        if subj[0] and obj[0] and subj[0] != obj[0]:
            # print(subj,obj)
            assign_row = subj[1] if subj[0] == 'row' else obj[1]
            assign_col = obj[1] if obj[0] == 'col' else subj[1]
            # print(assign_row, assign_col)
            table[assign_row][assign_col] = False if contain_negation else True
            # print(table)
    return table


def assign_table_with_qa(table, question, answers, rows, columns):
    col_ent_occur = []
    row_ent_occur = []
    match_pattern = '([A-Za-z0-9 ]+, ){1,}([a-zA-Z0-9 ]*)'
    # extract question and extract possible columns based on range "Monday to Friday"
    modified_question, possible_columns = analyze_question(question)
    contain_negation = False
    for wd in negation_words:
        if wd in modified_question:
            contain_negation = True

    if len(possible_columns) >= 1:
        possible_columns = possible_columns[0]
    if rows != columns:
        for col in columns:
            if col in question:
                col = col.replace('$', '\$')
                res = re.search(col, question)
                # print(col,question,res)
                col_ent_occur.append({'span': col, 'start': res.span()[0]})
        sorted_col = sorted(col_ent_occur, key=lambda k: k['start'])
        for row in rows:
            if row in question:
                res = re.search(row, question)
                row_ent_occur.append({'span': row, 'start': res.span()[0]})
        sorted_row = sorted(row_ent_occur, key=lambda k: k['start'])

        if len(sorted_col) > len(sorted_row):
            sorted_ent = sorted_col
        else:
            sorted_ent = sorted_row
    else:
        sorted_ent = [{'span': tmp} for tmp in list(orders.keys())]

    # order_flag = any([tmp in list(orders.keys()) for tmp in sorted_ent])
    option_based_tables = []
    for answer in answers:
        ans_table = copy.deepcopy(table)
        # print('the question and answer are: ', question, answer)
        possible_assignment = analyze_option_with_pattern(answer, rows, columns, sorted_ent, possible_columns)
        if not possible_assignment:
            ans_table, functions = match_qa_func_and_modify_table(modified_question, answer, rows, columns, ans_table)
        else:
            ans_table = modify_table_with_assignment(ans_table, possible_assignment, contain_negation)

        option_based_tables.append(ans_table)
    return option_based_tables

def find_assgiments_for_qa(table, question, answers, rows, columns):
    col_ent_occur = []
    row_ent_occur = []
    match_pattern = '([A-Za-z0-9 ]+, ){1,}([a-zA-Z0-9 ]*)'
    # extract question and extract possible columns based on range "Monday to Friday"
    modified_question, possible_columns = analyze_question(question)
    contain_negation = False
    for wd in negation_words:
        if wd in modified_question:
            contain_negation = True
    # v = False if contain_negation else True
    if len(possible_columns) >= 1:
        possible_columns = possible_columns[0]
    if rows != columns:
        for col in columns:
            if col in question:
                col = col.replace('$', '\$')
                res = re.search(col, question)
                # print(col,question,res)
                col_ent_occur.append({'span': col, 'start': res.span()[0]})
        sorted_col = sorted(col_ent_occur, key=lambda k: k['start'])
        for row in rows:
            if row in question:
                res = re.search(row, question)
                row_ent_occur.append({'span': row, 'start': res.span()[0]})
        sorted_row = sorted(row_ent_occur, key=lambda k: k['start'])

        if len(sorted_col) > len(sorted_row):
            sorted_ent = sorted_col
        else:
            sorted_ent = sorted_row
    else:
        sorted_ent = [{'span': tmp} for tmp in list(orders.keys())]

    # order_flag = any([tmp in list(orders.keys()) for tmp in sorted_ent])
    option_based_assignments = []
    option_functions = []
    for answer in answers:
        functions = []
        ans_table = copy.deepcopy(table)
        # print('the question and answer are: ', question, answer)
        possible_assignment,backup_flag = analyze_option_with_pattern(answer, rows, columns, sorted_ent, possible_columns,modified_question)
        if not possible_assignment or (possible_assignment and backup_flag):
            qa_assignments,functions = match_qa_func_and_modify_table(modified_question.strip(), answer, rows, columns, ans_table)
            if not qa_assignments and not functions:
                qa_assignments = [[]]
                if possible_assignment and backup_flag:
                    for assign in possible_assignment:
                        subj, obj = assign
                        qa_assignments[-1].append({'row': subj, 'column': obj, 'value': True})
                else:
                    row_ent_occur, col_ent_oocur = extract_par_col(question + ' ' + answer, rows, columns)
                    for r in row_ent_occur:
                        for c in col_ent_oocur:
                            qa_assignments[-1].append({'row':r['span'], 'column':c['span'],'value':True})

            # ans_table, functions = match_qa_func_and_modify_table(question, answer, rows, columns, ans_table)
        elif possible_assignment and not backup_flag:
            qa_assignments = [[]]
            for assign in possible_assignment:
                subj, obj = assign
                if subj[0] and obj[0] and subj[0] != obj[0]:
                    # print(subj,obj)
                    assign_row = subj[2] if subj[0] == 'row' else obj[2]
                    assign_col = obj[2] if obj[0] == 'col' else subj[2]
                    qa_assignments[-1].append({'row':assign_row,'column':assign_col,'value':True})
                    # print(assign_row, assign_col)
            # qa_assignments = modify_table_with_assignment(possible_assignment, contain_negation)
        if not qa_assignments and backup_flag:
            qa_assignments = [[]]
            for assign in possible_assignment:
                subj, obj = assign
                qa_assignments[-1].append({'row': subj, 'column': obj, 'value': True})

        option_functions.append(functions)
        option_based_assignments.append(qa_assignments)

    # print('option based assignment',option_based_assignments)
    return option_based_assignments, option_functions
def extract_par_col(text,rows,columns):
    col_ent_occur,row_ent_occur = [], []
    for col in columns:
        if col in text:
            col = col.replace('$', '\$')
            res = re.search(col, text)
            # print(col,question,res)
            col_ent_occur.append({'span': col, 'start': res.span()[0]})
    sorted_col = sorted(col_ent_occur, key=lambda k: k['start'])
    for row in rows:
        if row in text:
            res = re.search(row, text)
            row_ent_occur.append({'span': row, 'start': res.span()[0]})
    sorted_row = sorted(row_ent_occur, key=lambda k: k['start'])
    return sorted_row,sorted_col

def rank_entities(entities):
    quantity_ents = []
    rest_ents = []
    flag = False
    for group in [RP.orders, RP.numbers, RP.week_words, RP.month_words]:
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

def fact_in_question(question):
    ifflag, if_flag, if_pos , then_pos,question = extract_if_then(question)
    # print(flag,subsen_if,subsen_then)
    if if_flag:
        return question[if_pos+len('If '):then_pos]#if_pos,then_pos#subsen_if#,subsen_then
    else:
        return None#,None
import rule_update_table
from importlib import import_module
def derive_rules( rule, rows, columns,neural_func=None):
    # rule = rule.replace('later','after')
    functions = []
    if ', so does' in rule:
        rule = rule.replace(', so does',' so does')
    rule = rule.strip()
    ifflag, ifthenflag, ifpos, thenpos,rule = extract_if_then(rule)
    if_func, then_func = [],[]
    # print(rule)
    words = nltk.word_tokenize(rule)
    tags = nltk.pos_tag(words)
    func2pos, pos2func = match_words_tags_func(rule, words, tags)
    neg_count = 0
    for wd in words:
        if wd in negation_words:
            neg_count += 1
    negation = True if neg_count % 2 == 1 else False
    results = [(words[i], tags[i], pos2func[i]) for i in range(len(words))]
    # print(rule,words)
    t2c, c2t = map_charid_to_token_id(rule, words)

    ifpos = c2t[ifpos]
    thenpos = c2t[thenpos-1]
    par_occur, col_occur, func_name2pos = find_par_col_occur(rule, c2t, results, rows, columns)
    # print(par_occur,col_occur,func_name2pos)
    all_positions = []
    for func, positions in func_name2pos.items():#derive position for each function
        positions_pair = []
        for i,pos in enumerate(positions):
            if i==0:
                positions_pair.append(pos)
            else:
                if pos[0]-positions_pair[-1][-1]==1:
                    positions_pair[-1].extend(pos)
                else:
                    positions_pair.append(pos)
        for pos in positions_pair:
            if par_occur or col_occur:

                funcs = getattr(API, 'func_{}'.format(func))(par_occur, col_occur, words, pos,
                                                                              original_assignments, negation, rows,columns)
                # functions.extend(funcs)
                # print('functions: ',funcs)
                for tmp in funcs:
                    all_positions.append(pos)
                # funcs = merge_funcs_with_or(funcs,words)
                if ifthenflag:
                    ifpart = list(range(ifpos,thenpos))
                    thenpart = list(range(thenpos,len(rule)))
                    if min(pos) in ifpart or max(pos) in ifpart:
                        if_func.extend(funcs)
                    elif min(pos) in thenpart or max(pos) in thenpart:
                        then_func.extend(funcs)
                else:
                    functions.extend(funcs)
    if ifflag:
        functions = merge_funcs_with_iff(functions,words,ifpos,thenpos,all_positions)
    elif 'neither' in words:
        functions = merge_funcs_with_neither(functions,words,all_positions)
    elif 'or' in words:
        functions = merge_funcs_with_or(functions, words,all_positions)
    else:
        functions = merge_funcs_with_and(functions,words,all_positions)

    functions = merge_funcs_with_unless(functions,words,all_positions)
    if ifthenflag and if_func and then_func:
        # print(if_func)
        # print(then_func)
        ifthenrule = getattr(API, 'func_if_then')(if_func, then_func, rows, columns)
        # print(ifthenrule)
        functions.extend(ifthenrule)

    return functions

def process_order_word(text,all_cols):
    if any([wd in RP.orders.keys() for wd in all_cols]):
        text=text.replace('last',all_cols[-1])
    return text

def process_text(text,all_pars,all_cols):
    text = process_order_word(text,all_cols)
    return text

def analyze_question_type(tags,rows,columns):
    question_type = None
    for tag in tags:
        if 'ordering' in tag:
            question_type = 'ordering'
        elif 'grouping' in tag:
            question_type = 'grouping'
    if not question_type:
        if any([(col in all_num_words or col.isnumeric()) for col in columns]):
            question_type = 'ordering'
            # return question_type
    if not question_type:
        question_type = 'grouping'
        return question_type
    if len(rows)>len(columns) :
        question_type = 'grouping'
    return question_type





def print_table(table, rows, columns):
    print('\t', columns)
    # print(table)
    for idx, row in enumerate(rows):
        print(row, table[idx])
from collections import Counter
def transfer_rule_result(rule_res):
    ruleid2funcs = {}
    ruleid2rule = {}
    funcset = []
    for ins in rule_res:
        # print(ins)
        label = ins['labels'][0]
        rule_id = '{}_{}'.format(ins['doc_id'],ins['rule_id'])
        kw = ins['keywords'][0]
        rule=ins['rule']
        funcset.append(label)
        if label!='NULL':
            func = {'function':label,'arg1':{'text':kw['text_of_key1'],'position':(kw['begin_of_key1'],kw['end_of_key1'])},
                    'arg2':{'text':kw['text_of_key2'],'position':(kw['begin_of_key2'],kw['end_of_key2'])}}

            if rule_id not in ruleid2funcs.keys():
                ruleid2funcs[rule] = []
                ruleid2rule[rule] = ins['rule']

            ruleid2funcs[rule].append(func)


    return ruleid2funcs,ruleid2rule
doc_visit = []
ans2label = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
basic_dir = '../data'
# basic_dir = 'D:\PycharmProject\LSAT_rule\data'
# rule_res_p = os.path.join(basic_dir, 'neural-rule-extraction/ar_val_rule_extraction_88.41%.json')
datap = os.path.join(basic_dir, 'new_data/ar_test_analyze_condition.json')
# cp_dp_ner_path = os.path.join(basic_dir, 'AR_DevelopmentData_cp_ner_dp_results.json')
outp = os.path.join(basic_dir, 'new_data/ar_test_modify_context.json')
# cp_dp_ner_results = json.load(open(cp_dp_ner_path, 'r'))
# rule_res = json.load(open(rule_res_p,'r'))
instances = json.load(open(datap, 'r'))
modified_instances = []
doc_index = 0
doc_visit.append(instances[0]['context'])
all_output = [{'context':instances[0]['context'],'questions':[]}]
all_label = []
all_predict = []
sentence_count=0
# ruleid2funcs, ruleid2rule = transfer_rule_result(rule_res)

for ins_idx, instance in enumerate(instances):
    context = instance['context']

    if (context in doc_visit) == False:
        doc_visit.append(context)
        doc_index += 1
        all_output.append({'context':instance['context'],'questions':[]})
        sentence_count = 0

    sentence_count+=1

    sentences = nltk.sent_tokenize(context)
    all_sentence = []
    for sen in sentences:
        all_sentence.extend(nltk.sent_tokenize(sen))
    # sentences = cp_dp_ner_results[doc_index]['sentences']
    # sentence_cp_results = cp_dp_ner_results[doc_index]['sentence_cp_results']

    #read information: including participants and positions
    question = instance['question']
    answers = instance['answers']
    label = instance['label']
    all_label.append(label)
    id_string = instance['id_string']
    #rows indicate participants, columns indicate positions
    rows = instance['rows']
    columns = instance['columns']

    #rank participants and columns based on predefined orders: like first, second, third etc.
    columns = rank_entities(columns)
    rows = rank_entities(rows)
    #if participants are the same with positions, substitude positions to orders, like (A,B) -> (first, second)
    if rows == columns:
        columns = [list(RP.orders.keys())[idx] for idx in range(len(rows))]

    #analyze the type of questions and process text
    question_type = analyze_question_type(instance['tags'],rows,columns)
    question = process_text(question,rows,columns)
    answers = [process_text(ans,rows,columns) for ans in answers]
    sentences = [process_text(sen,rows,columns) for sen in sentences]

    #extract fact and rules from sentences based on (text, participants, columns)
    fact_ids, rule_ids = extract_facts_rules(sentences, rows, columns)
    #extract fact (hypothetics) in the question
    inner_question_fact = fact_in_question(question)
    facts = [sentences[id] for id in fact_ids]
    if inner_question_fact:
        facts.append(inner_question_fact)
    rules = [sentences[id] for id in rule_ids]
    all_output[-1]['columns'] = columns
    all_output[-1]['rows'] = rows
    all_output[-1]['facts'] = facts
    all_output[-1]['rules'] = rules
    all_output[-1]['questions'].append({'question':question,'answers':answers,'type':instance['tags']})

    #extract initial assignment by fact
    original_assignments, add_rules = assign_table_with_facts(rows, columns, facts)
    #if the initial assignment can not be found, take the fact as a rule
    rules = rules + add_rules
    # rule_cp_results = [sentence_cp_results[idx] for idx in rule_ids]

    #extract logical functions from the rules
    all_rule_func = []
    # print(
    for rule_id, rule in enumerate(rules):

        rid = '{}_{}'.format(doc_index,rule_id)
        # if rule in ruleid2funcs.keys():
        #     neural_funcs = ruleid2funcs[rule]
        # else:
        #     neural_funcs  = []
        rule_func = derive_rules(rule,rows,columns)#,neural_funcs)
        all_rule_func.extend(rule_func)

    #construct rule tree based on extracted functions
    rule_tree = RuleTree(rows,columns,all_rule_func,original_assignments,question_type)
    leaf_nodes, each_level_node = rule_tree.obtain_root_and_construct_tree()
    #all the leaf node (possible assignments)
    # print('all the possible solution: {}'.format(len(leaf_nodes)))
    # for lid,level in enumerate(each_level_node):
    #     print(lid+1,len(level))

    #extract option-based assignments and functions from each option
    option_based_assignments,option_functions = find_assgiments_for_qa(original_assignments,question,answers,rows,columns)#assign_table_with_qa(table,question,answers,rows,columns)
    option_functions = merge_option_functions(option_functions)

    #choose question type and calculate scores for each option
    Question = choose_question_type(question,leaf_nodes,rows,columns,question_type,instance['tags'],all_rule_func)#QuestionType(question,leaf_nodes)
    sorted_score = Question.select_answers(answers,option_based_assignments,option_functions)
    # print(sorted_score)

    max_score, prediction = sorted_score[0]
    if all(x[0] == sorted_score[0][0] for x in sorted_score):#all([tmp[0]==0 for tmp in sorted_score]):
        prediction = 0

    all_predict.append(prediction)
    # print(max_score,prediction,label,doc_index,sentence_count)

compare = [all_predict[idx] == all_label[idx] for idx in range(len(all_label))]
print('Overall precision',sum(compare) / len(compare))

#
# with open(outp,'w',encoding='utf8') as outf:
#     json.dump(all_output,outf,indent=4)
