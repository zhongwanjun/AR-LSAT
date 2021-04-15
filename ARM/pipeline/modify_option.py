import json
import os
import nltk
import re
from tqdm import tqdm
import sys
from rule_pattern import RulePattern
from collections import Counter
RP = RulePattern()
# list(number_of_exams) -> dict ('sections') (list of sections,select 0) -> dict(passages)
# -> list (number of passages) -> dict(['id',
# 'questions', 'fatherId', 'passageId', 'passage', 'passageWithLabel'])

orders = {
            'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'fifth': 5, 'sixth': 6, 'seventh': 7,
            'eighth': 8, 'ninth': 9, 'tenth': 10, 'eleventh': 11, 'twelfth': 12,'thirteenth':13,
            'fourteenth':14,'fifteenth':15,'sixteenth':16
        }
week_words = {'Monday':1, 'Tuesday':2, 'Wednesday':3, 'Thursday':4, 'Friday':5, 'Saturday':6, 'Sunday':7}
month_words = {
"January":1, 'February':2, 'March':3, 'April':4, 'May':5, 'June':6, 'July':7, 'August':8, 'September':9, 'October':10,
    'November':11, 'December':12
}
def clean_doc(doc):
    doc = ' '+doc
    doc = doc.replace('—', ' — ')
    doc = doc.replace('—', ' — ')
    doc = doc.replace('\u2014',' ')
    return doc


def analyze_question(question):
    attention_word = ['then','which','what']
    useful_parts = []
    for attn_word in attention_word:
        if attn_word in question:
            useful_parts.append(question[question.find(attn_word)+len(attn_word):])
        else:
            useful_part = question
    useful_parts = sorted(useful_parts,key=lambda k:len(k),reverse=True)
    if useful_parts:
        if (len(useful_parts[0])<len(question)):
            useful_part = useful_parts[0]

    possible_columns = extract_quantity(question)
    return useful_part,possible_columns

def rewrite(parts,rewrite_list):
    # print(parts,rewrite_list)
    for idx in range(len(parts)):
        if 'and' in parts[idx]:
            tmp = parts[idx].replace('and', '')
            parts[idx] = ' and the {} is {}'.format(rewrite_list[idx], tmp)
        else:
            parts[idx] = 'the {} is {}'.format(rewrite_list[idx], parts[idx])
    return parts

def extract_quantity(text):
    range_results = re.finditer(RP.range_pattern,text)
    range_quantity = []
    for match in range_results:
        range_quantity.append({'text':match.group() ,'span':match.span()})

    all_possible_columns = []
    for rang in range_quantity:
        nums = re.finditer(RP.number_pattern,rang['text'])
        all_nums = []
        num_dict = None
        start, end = 9999, -1
        for num in nums:
            # print(num)
            start = min(start, num.span()[0])
            end = max(end, num.span()[1])
            num, num_dict = RP.mapnum2int(num.group())
            all_nums.append(num)
            num_dict = num_dict

        min_num, max_num = min(all_nums), max(all_nums)
        columns = []
        if num_dict:
            for item in num_dict.items():
                if item[1] >= min_num and item[1] <= max_num and item[0].lower() not in columns:
                    columns.append(item[0])
        else:
            columns = list(range(min_num, max_num + 1))
        all_possible_columns.append(columns)
    all_possible_columns = sorted(all_possible_columns,key=lambda k: len(k))
    return all_possible_columns


def modify_option_based_on_pattern(question,answers,rows,columns):
    col_ent_occur = []
    row_ent_occur = []
    modified_options = []
    match_pattern = '([A-Za-z0-9 ]+, ){1,}([a-zA-Z0-9 ]*)'
    question,possible_columns = analyze_question(question)
    if len(possible_columns)>=1:
        possible_columns = possible_columns[0]
    type = 'columns'
    if rows!=columns:
        for col in columns:
            if col in question:
                col = col.replace('$','\$')
                res = re.search(col,question)
                # print(col,question,res)
                col_ent_occur.append({'span':col,'start':res.span()[0]})
        sorted_col = sorted(col_ent_occur,key=lambda k : k['start'])
        for row in rows:
            if row in question:
                res = re.search(row,question)
                row_ent_occur.append({'span':row,'start':res.span()[0]})
        sorted_row = sorted(row_ent_occur,key=lambda k : k['start'])

        if len(sorted_col) > len(sorted_row):
            sorted_ent = sorted_col
            type = 'columns'
        else:
            sorted_ent = sorted_row
            type = 'rows'
    else:
        sorted_ent = [{'span':tmp} for tmp in list(orders.keys())]
        type = 'order'

    order_flag = any([tmp in list(orders.keys()) for tmp in sorted_ent])

    for option in answers:
        all_res = re.finditer(match_pattern,option)
        for res in all_res:
            parts = res.group().split(', ')

            if len(parts)<=len(sorted_ent):
                parts = rewrite(parts,[tmp['span'] for tmp in sorted_ent])
            elif possible_columns and len(parts)<=len(possible_columns):
                    parts = rewrite(parts,possible_columns)
            elif len(sorted_ent)==1 and order_flag==False:
                parts = rewrite(parts, [sorted_ent[0]['span'] for i in range(len(parts))])
            else:
                parts = rewrite(parts, list(orders.keys()))

            replace_str = ', '.join(parts)
            option = option.replace(res.group(),replace_str)
        modified_options.append(option)
    return modified_options

def modify_option(options):
    match_pattern = '([A-Za-z0-9 ]+, ){1,}([a-zA-Z0-9 ]*)' #'([A-Za-z0-9]+, ){1,}([a-zA-Z0-9]*)'
    modified_options=[]
    for option in options:
        all_res = re.finditer(match_pattern,option)
        for res in all_res:
            parts = res.group().split(', ')
            for idx in range(len(parts)):
                if 'and' in parts[idx]:
                    tmp = parts[idx].replace('and','')
                    parts[idx] = ' and the {} is {}'.format(list(orders.keys())[idx],tmp)
                else:
                    parts[idx] = 'the {} is {}'.format(list(orders.keys())[idx],parts[idx])
            replace_str = ', '.join(parts)
            option = option.replace(res.group(),replace_str)
        modified_options.append(option)
    return modified_options


def replace_participants(context,par2ent,participants):

    for par in participants:
        context = context.replace(par,par2ent[par])

    return context

'''
ans2label = {'A':0,'B':1,'C':2,'D':3,'E':4}
basic_dir = '../data'
datap = os.path.join(basic_dir, 'model_data/ar_val_analyze_condition.json')
outp =  os.path.join(basic_dir, 'modified_model_data/ar_val_modify_context.json')
instances = json.load(open(datap, 'r'))
modified_instances = []
for ins_idx, instance in enumerate(instances):
    context = instance['context']

    question = instance['question']
    answers = instance['answers']
    label = instance['label']
    id_string = instance['id_string']
    rows = instance['rows']
    columns = instance['columns']

    modified_answers = modify_option_based_on_pattern(question,answers,rows,columns)
    # ent_for_replace = ['Alice', 'Bob', 'Charles', 'Daniel', 'Eugene', 'Frank', 'Gary',
    #                    'Henry', 'Icey', 'Joe', 'Keith', 'Louis', 'Marie', 'Neko']
    # par2ent = {}
    # for i in range(len(rows)):
    #     par2ent[rows[i]] = ent_for_replace[i]
    # new_participants = list(par2ent.values())

    # modified_context = replace_participants(context,par2ent,rows)
    # modified_question = replace_participants(context,par2ent,rows)
    # modified_answers = [replace_participants(ans,par2ent,rows) for ans in modified_answers]
    # instances[ins_idx]['modified_context'] = context
    # instances[ins_idx]['modified_question'] = question
    # instances[ins_idx]['modified_rows'] = new_participants
    instances[ins_idx]['modified_answers'] = modified_answers
    # print(context,modified_context)
    # print(question, modified_question)
    # print(answers, modified_answers)
    # print('-------------------------------')
    # input()
    # if modified_answers!=answers:
    #     print('context:',context)
    #     print('question: ',question)
    #     print('answer',answers)
    #     print('modified_context',modified_answers)
    #     print('participants',rows)
    #     print('assignment',columns)
    #     input()



with open(outp,'w',encoding='utf8') as outf:
    json.dump(instances,outf)

'''



