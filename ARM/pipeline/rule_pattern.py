import re
import inflect
import nltk
import numpy as np
import json
# from allennlp.predictors.predictor import Predictor
# import allennlp_models.tagging
# import allennlp_models.structured_prediction
class RulePattern():

    def __init__(self):
        super(RulePattern, self).__init__()
        numbers = {
            #'an':1,
            'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 'eleven': 11,'twelve': 12
        }
        orders = {
            'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'fifth': 5, 'sixth': 6, 'seventh': 7,
            'eighth': 8, 'ninth': 9, 'tenth': 10, 'eleventh': 11, 'twelfth': 12,'thirteenth':13,
            'fourteenth':14,'fifteenth':15,'sixteenth':16,'last':17,
        }
        simple_order = {
            '1st':1, '2nd':2,'3rd':3,'4th':4,'5th':5,'6th':6,'7th':7,'8th':8,'9th':9
        }
        self.numbers = numbers.copy()
        for num in numbers.items():
            self.numbers[num[0].capitalize()] = num[1]
        self.orders = orders.copy()
        self.simple_order = simple_order.copy()
        for num in orders.items():
            self.orders[num[0].capitalize()] = num[1]
        self.pos_useful_tag = ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
                               'V', 'PDT', 'PRP', 'RBR', 'RBS']
        self.entity_tag = ['NN', 'NNS', 'NNP', 'NNPS']
        self.week_words = {'Monday':1, 'Tuesday':2, 'Wednesday':3, 'Thursday':4, 'Friday':5, 'Saturday':6, 'Sunday':7}
        self.month_words = {
        "January":1, 'February':2, 'March':3, 'April':4, 'May':5, 'June':6, 'July':7, 'August':8, 'September':9, 'October':10,
            'November':11, 'December':12
        }
        self.num_words = ['first','second','three','four','five','six','seven','']
        self.no_use_single_words = ['The', 'She', 'He', 'They', 'It', 'Them', 'Their', 'A', 'On', 'In', 'To', 'Where',
                                    'There','Each','If','following','Neither','Except']

        self.all_num_dict = [self.numbers,self.orders,self.simple_order,self.week_words,self.month_words]
        self.quantity_words = ['once','each','twice','a']
        self.keywords = ['circular']
        number_joint = '|'.join(list(self.numbers.keys()))
        self.entity_patterns = ['(([0-9:$,]+)|((a|an|the|one|two|three|four|five|six|seven|eight|night|ten|[0-9]+| )*([a-zA-Z0-9 ]{1,})( [0-9])*), ){2,}(and|or)*(a|an|the|one|two|three|four|five|six|seven|eight|night|ten|[0-9]+| )*(([a-zA-Z0-9:. ]{1,}(( [0-9])*))|([$,0-9]+( [a-z]+)))',#'(([0-9:$,]+)|((a|an|the|one|two|three|four|five|six|seven|eight|night|ten|[0-9]+| )*([a-zA-Z0-9. ]{1,})( [0-9])*), ){2,}(and|or)*(a|an|the|one|two|three|four|five|six|seven|eight|night|ten|[0-9]+| )*(([a-zA-Z0-9:. ]{1,}(( [0-9])*))|([$,0-9]+( [a-z]+)))',#(([0-9:$,]+)|((a|an|the|one|two|three|four|five|six|seven|eight|night|ten|[0-9]+| )*([a-zA-Z0-9.]{1,})( [0-9])*), ){2,}(and|or)*(a|an|the|one|two|three|four|five|six|seven|eight|night|ten|[0-9]+| )*(([a-zA-Z0-9:. ]{1,}(( [0-9])*))|([$,0-9]+( [a-z]+)))',
                               '(([0-9:$,]+)|((a|an|the|one|two|three|four|five|six|seven|eight|night|ten|[0-9]+| )*([a-zA-Z0-9. ]{1,})( [0-9])*), ){2,}(and|or)*(a|an|the|one|two|three|four|five|six|seven|eight|night|ten|[0-9]+| )*(([a-zA-Z0-9:. ]{1,}(( [0-9])*))|([$,0-9]+( [a-z]+)))',
                               # '([A-Za-z0-9]+, ){2,}and ([0-9A-Za-z ]+)'
                               '([A-Za-z ]+, ){2,}and ([A-Za-z ]+)',
                                '(— |: )([A-Za-z ]+) (and|or) ([A-Za-z ]+)']#higher participant performance
            #'((a [0-9:$,]+)|((a|an|the|one|two|three|four|five|six|seven|eight|night|ten|[0-9]+| )*([a-zA-Z0-9.]{1,})( [0-9])*), ){2,}(and|or)*(a|an|the|one|two|three|four|five|six|seven|eight|night|ten|[0-9]+| )*(([a-zA-Z0-9:]{1,}( [0-9])*)|([$,0-9]+( [a-z]+)))'#higher col performance
        #'(([0-9:$,]+)|((a|an|the|one|two|three|four|five|six|seven|eight|night|ten|[0-9]+| )*([a-zA-Z0-9.]{1,})( [0-9])*), ){2,}(and|or)*(a|an|the|one|two|three|four|five|six|seven|eight|night|ten|[0-9]+| )*[a-zA-Z0-9.]{1,}( [0-9])*'
            #'(([0-9:]+)|((a|an|the|one|two|three|four|five|six|seven|eight|night|ten|[0-9]+| )*([a-zA-Z0-9.]{1,})( [0-9])*), ){2,}(and|or)*(a|an|the|one|two|three|four|five|six|seven|eight|night|ten|[0-9]+| )*[a-zA-Z0-9.]{1,}( [0-9])*'
        #'((a|an|the|one|two|three|four|five|six|seven|eight|night|ten|[0-9]+| )*([a-zA-Z0-9]{1,})( [0-9])*, ){2,}(and|or)*(a|an|the|one|two|three|four|five|six|seven|eight|night|ten|[0-9]+| )*[a-zA-Z0-9]{1,}( [0-9])*'
        self.quantity_pattern = '( |— )(|{number}|[0-9]+)+ ([a-zA-Z]+)[ —.,](({number}|[0-9])+[ ,.])*'.format(number=number_joint)
        self.range_pattern = '[a-zA-Z ]*({order}|{number}|{week}|([0-9]+))+ (\([a-z]+\) )*(through|to) [a-zA-Z ]*({order}|{number}|{week}|([0-9]+.))+'.format(
            number=number_joint,order='|'.join(list(orders.keys())+list(simple_order.keys())),week='|'.join(list(self.week_words.keys())),month='|'.join(list(self.month_words.keys())))
        self.number_pattern = '({simple_order}|{order}|{number}|{week}|{month}|([0-9]+))+'.format(number=number_joint,simple_order='|'.join(list(simple_order.keys())),order='|'.join(list(self.orders.keys())),week='|'.join(list(self.week_words.keys())),month='|'.join(list(self.month_words.keys())))
        #'((a|an|the|one|two|three|four|five|six|seven|eight|night|ten|[0-9]+| )*([a-zA-Z0-9]{1,}), ){2,}(and|or)*(a|an|the|one|two|three|four|five|six|seven|eight|night|ten|[0-9]+| )*[a-zA-Z0-9]{1,}'
        # self.predictor_ner = Predictor.from_path(
        #     "https://storage.googleapis.com/allennlp-public-models/ner-model-2020.02.10.tar.gz",cuda_device=0)
        # self.predictior_cp = Predictor.from_path(
        #     "https://storage.googleapis.com/allennlp-public-models/elmo-constituency-parser-2020.02.10.tar.gz",cuda_device=0)
        # self.predictior_dp = Predictor.from_path('https://storage.googleapis.com/allennlp-public-models/biaffine-dependency-parser-ptb-2020.04.06.tar.gz',cuda_device=0)
        # self.openie_predictor = Predictor.from_path(
        #     "https://s3-us-west-2.amazonaws.com/allennlp/models/openie-model.2018-08-20.tar.gz")
        # self.srl_predictor = Predictor.from_path(
        #     "https://s3-us-west-2.amazonaws.com/allennlp/models/srl-model-2018.05.25.tar.gz")
        self.stemm = nltk.PorterStemmer()
        self.tokenizer = nltk.word_tokenize
        self.inflect = inflect.engine()


    def modify_option(self,options):
        match_pattern = '([A-Za-z0-9]+, ){1,}([a-zA-Z0-9]*)'
        modified_options=[]
        for option in options:
            all_res = re.finditer(match_pattern,option)
            for res in all_res:
                parts = res.group().split(', ')
                for idx in range(len(parts)):
                    if 'and' in parts[idx]:
                        parts[idx] = ' and the {} is'.format(list(self.orders.keys())[idx])
                    else:
                        parts[idx] = 'the {} is {}'.format(list(self.orders.keys())[idx],parts[idx])
                replace_str = ', '.join(parts)
                option = option.replace(res.group(),replace_str)
            modified_options.append(option)
        return modified_options


    def find_wspan(self,text,tokens,cspan,start_idx):
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
        return (start_idx+cand[0][0],start_idx+cand[0][1]+1)


    def extract_entity_pattern(self,text, tokens,sen_start_idx):

        ent_groups = []
        all_ent = []
        repeat_group = []
        for idx,entity_pattern in enumerate(self.entity_patterns):
            entity_parts = re.finditer(entity_pattern, text)
            if entity_parts:
                for entity_part in entity_parts:
                    tmp = str(entity_part.group()).replace(', ', '#')
                    # print(tmp)
                    tmp = str(tmp).replace('and ', '#')
                    tmp = str(tmp).replace('or ', '#')
                    tmp = tmp.split('#')
                    while '' in tmp:
                        tmp.remove('')
                    count = 0
                    # print(tmp)
                    for widx,item in enumerate(tmp):
                        if self.exist(item,all_ent):
                            count+=1
                        item = item.replace('—','').strip()
                        if len(item.split(' '))>4:#if length of some word larger than 4,drop this instance
                            count+=len(tmp)
                    if count>=(2/3)*len(tmp):#if the number of entities in current group already found larger than 2/3 of the total entities in the group, continue
                        repeat_group.append({'entities': tmp,
                                           'span': self.find_wspan(entity_part.group(), tokens, entity_part.span(),
                                                                   sen_start_idx)})
                        continue
                    else:
                        ent_groups.append({'entities': tmp,
                                           'span': self.find_wspan(entity_part.group(), tokens, entity_part.span(),
                                                                   sen_start_idx)})


                    all_ent.extend(tmp)
            # sorted_repeat_group = sorted(repeat_group,key=lambda k:len(k['entities']),reverse=True)

            # if idx>0 and ent_groups:

            if len(ent_groups)>1:
                break

        return ent_groups


    def extract_quantity(self, text, tokens,start_idx):
        results = re.finditer(self.quantity_pattern,text)
        re_quantity = []
        for match in results:
            re_quantity.append({'text':match.group().strip(' ,.—') ,'span':self.find_wspan(match.group(),tokens, match.span(),start_idx)})
        range_results = re.finditer(self.range_pattern,text)
        range_quantity = []
        for match in range_results:
            range_quantity.append({'text':match.group().strip(' ,.—') ,'span':self.find_wspan(match.group(),tokens, match.span(),start_idx)})
        return re_quantity,range_quantity

    def match_number_entity(self,quantity,ent_groups):
        for qua_idx,qua in enumerate(quantity):
            qua_pos = qua['span']
            min_dis, min_idx = 9999,None
            for gidx,group in enumerate(ent_groups):
                group_pos = group['span']
                dis = group_pos[0] - qua_pos[1]
                if qua_pos[0] in list(range(group_pos[0],group_pos[1])):
                    dis = 0
                if (dis>=0 and dis<min_dis):
                    min_idx = gidx
                    min_dis = dis

            quantity[qua_idx]['match_entity_group'] = ent_groups[min_idx] if (ent_groups and min_idx!=None) else None
        return quantity

    def extract_participants(self, leaf_nodes):
        nouns = []
        quantity = []
        numbers = []
        for idx, node in enumerate(leaf_nodes):
            #calculate nouns and quantity
            flag = False
            if node['nodeType'] in self.entity_tag:
                #middle of the sequence
                if idx != 0 and idx != len(leaf_nodes) - 1:
                    if (leaf_nodes[idx - 1]['word'] in ['—', ',', 'and']) or (
                            leaf_nodes[idx + 1]['word'] in [',']):
                        flag = True
                    elif leaf_nodes[idx-1]['nodeType']=='CD':
                        quantity.append((leaf_nodes[idx-1]['word'],leaf_nodes[idx]['word'],idx))
                #start of sequence
                elif idx == 0:
                    if (leaf_nodes[idx + 1]['word'] in [',']):
                        flag = True
                #end of the sequence
                elif idx == len(leaf_nodes) - 1:
                    if (leaf_nodes[idx - 1]['word'] in ['—', ',', 'and']):
                        flag = True
                    elif leaf_nodes[idx-1]['nodeType']=='CD':
                        quantity.append((leaf_nodes[idx-1]['word'],leaf_nodes[idx]['word'],idx))
                if flag==False:
                    if node['word'] in list(self.month_words.keys())+list(self.week_words.keys()):
                        if idx==0:
                            quantity.append(('null',node['word'],idx))
                        else:
                            quantity.append((leaf_nodes[idx - 1]['word'], leaf_nodes[idx]['word'], idx))
                        flag = True
                if flag:
                    nouns.append((node['word'],idx))
            #if the word is number, find useful sorrounding context
            if node['nodeType']=='CD' or (node['word'].lower() in self.quantity_words):
                start=0 if idx==0 else idx-1
                for j in range(idx,0):
                    if leaf_nodes[j]['nodeType'] in list(self.entity_tag)+['CD'] or leaf_nodes[j]['word'] in [',','.','—']:
                        start = j
                        break
                for j in range(idx+1,len(leaf_nodes)):
                    if leaf_nodes[j]['nodeType'] in list(self.entity_tag)+['CD'] or leaf_nodes[j]['word'] in [',','.','—']:
                        end = j
                        break

                numbers.append(([leaf['word'] for leaf in leaf_nodes[start:end+1]],node['word'],idx))

        results = {'nouns':nouns, 'quantity':quantity,'numbers':numbers}
        return results

    def get_noun(self,tree,nps):
        if isinstance(tree, dict):
            if "children" not in tree:
                if (tree['nodeType'] in self.entity_tag):
                    nps.append(tree['word'])
                # if tree['nodeType'] in ["NNP","NN"]:
                    # print(tree['word'])
                    # print(tree)
                    # nps.append(tree)

            elif "children" in tree:
                # cls.get_NP(tree['children'], nps)
                if (tree['nodeType'] in self.entity_tag):
                    # print(tree['word'])
                    # nps.append(tree['word'])
                    self.get_noun(tree['children'], nps)
                else:
                    self.get_noun(tree['children'], nps)
        elif isinstance(tree, list):
            for sub_tree in tree:
                self.get_noun(sub_tree, nps)

        return nps

    def check_contain_num(self,tree):
        if isinstance(tree, dict):
            if "children" not in tree:
                if tree['nodeType'] == 'CD':
                    return True
                else:
                    return False
            else:
                return self.check_contain_num(tree['children'])
        elif isinstance(tree, list):
            ans = False
            for sub_tree in tree:
                ans = ans | self.check_contain_num(sub_tree)
            return ans

    def get_np_has_number(self,tree, nps, subtree):
        if isinstance(tree, dict):
            if "children" not in tree:
                if tree['nodeType'] == "NP":
                    # print(tree['word'])
                    # print(tree)
                    nps.append(tree['word'])
            elif "children" in tree:
                if tree['nodeType'] == "NP":
                    # print(tree['word'])
                    #                 print(tree['word'],check_contain_num(tree))
                    if self.check_contain_num(tree):
                        nps.append(tree['word'])
                        subtree.append(tree)

                #                     get_np_has_number(tree['children'], nps,subtree)
                else:
                    self.get_np_has_number(tree['children'], nps, subtree)
        elif isinstance(tree, list):
            for sub_tree in tree:
                self.get_np_has_number(sub_tree, nps, subtree)
        return nps, subtree

    def combine_cond_keywords(self,keywords):
        #combine keywords especially for condition extraction
        for i,group in enumerate(keywords['entity_group']):
            for j,ent1 in enumerate(group['entities']):
                ent1 = ent1.strip().replace(', or', '')
                keywords['entity_group'][i]['entities'][j] = ent1
                for k,ent2 in enumerate(keywords['entity']):
                    if ent1 in ent2:
                        keywords['entity_group'][i]['entities'][j]=ent2
        return keywords

    def combine_common_keywords(self,keywords):
        #combine keywords for common noun phrase extraction and entity extration
        for i,ent2 in enumerate(keywords['entity']):
            for j,useless_wd in enumerate(self.no_use_single_words):
                if (useless_wd+' ') in ent2:
                    keywords['entity'][i] = keywords['entity'][i].strip(useless_wd+' ')
        for i, ent1 in enumerate(keywords['noun']):
            for j, ent2 in enumerate(keywords['entity']):
                if ent1 in ent2:
                    keywords['noun'].pop(i)
                    break
        kws = list(set(keywords['noun']+keywords['entity']))
        for i,kw in enumerate(kws):
            if kw in self.no_use_single_words:
                kws.pop(i)
            if ',' in kws:
                tmp = kws[i]
                kws.pop(i)
                kws.extend(tmp.split(','))
        keywords['filtered_keywords'] = kws
        return keywords

    def extract_question_keywords(self, text, tokens, ent_res):
        keywords = {'noun': [], 'text':text, 'entity':[]}
        all_ents = self.extract_entity_allennlp(ent_res['words'], ent_res['tags'])
        keywords['entity'].extend(all_ents)
        tree = tokens['hierplane_tree']['root']
        nps = []
        nps = self.get_NP(tree,nps)
        keywords['noun'] = nps
        return keywords

    def extract_option_keywords(self, text, tokens, ent_res):
        return self.extract_question_keywords(text,tokens,ent_res)

    def found_key_words_common(self, claim, tokens, ent_res):
        key_words = {'noun': [], 'text': claim, 'entity':[]}
        tree = tokens['hierplane_tree']['root']
        nps, subtrees = [],[]
        nps, subtrees = self.get_np_has_number(tree,nps,subtrees)
        key_words['quantity_noun_phrase'] = nps
        # key_words['quantity_subtree'] = subtrees
        all_ents = self.extract_entity_allennlp(ent_res['words'], ent_res['tags'])
        key_words['entity'].extend(all_ents)
        nps = []
        tree = tokens['hierplane_tree']['root']
        nps = self.get_noun(tree, nps)
        key_words['noun'] = nps
        return key_words

    def mapnum2int(self,num):
        # print(num)
        for d in self.all_num_dict:
            if num in d.keys():
                return (d[num],d)

        return int(num),None

    def exist(self,item1, lst):
        for item2 in lst:
            if item1.strip() in item2.strip() or item2.strip() in item1.strip():
                return True

    def extract_columns_from_range(self,quantity,doc):

        nums = re.finditer(self.number_pattern,quantity)
        all_nums = []
        num_dict = None
        start,end=9999,-1
        print(nums)
        for num in nums:
            # print(num)
            start = min(start, num.span()[0])
            end = max(end, num.span()[1])
            num,num_dict = self.mapnum2int(num.group())
            all_nums.append(num)
            num_dict = num_dict
        min_num,max_num = min(all_nums),max(all_nums)
        columns = []
        if num_dict:
            for item in num_dict.items():
                if item[1]>=min_num and item[1]<=max_num and item[0].lower() not in columns:
                    columns.append(item[0])
        else:
            columns = [str(num) for num in range(min_num,max_num+1)]

        if start<9999 and end!=-1:
            modified_doc = doc.replace(quantity[start:end],', '.join(columns))
        else:
            modified_doc = doc
        return columns, modified_doc

    def found_key_words_leading_sen(self, claim,tokens, ent_res,sid):
        #extract entity and quantity especially for two leading sentences for condition extractions
        #use regex pattern to extract entities and quantity
        key_words = { 'text': claim, 'quantity_pairs': [], 'entity': []}
        nps = []

        # tree = tokens['hierplane_tree']['root']
        # leaf_nodes = self.get_noun(tree, nps)
        # results = self.extract_participants(leaf_nodes)
        # key_words['noun'].extend(results['nouns'])#.extend(results['quantity'])
        # key_words['numbers'] = results['numbers']

        all_ents = self.extract_entity_allennlp(ent_res['words'], ent_res['tags'])
        key_words['entity'].extend(all_ents)

        ent_groups = self.extract_entity_pattern(claim,tokens['tokens'],sid)
        key_words['entity_group'] = ent_groups

        # add_keywords = self.combine_cond_keywords(key_words)
        quantity,range_quantity = self.extract_quantity(claim,tokens['tokens'],sid)
        # print(claim, '\n')
        # print(tokens['tokens'])
        # print('quantity is:',quantity,'\n')
        # print('entity_group is: ',ent_groups,'\n')
        quantity = self.match_number_entity(quantity,ent_groups)
        range_quantity = self.match_number_entity(range_quantity,ent_groups)

        # print('matched quantity is :',quantity)
        # print('---------------------------------------')
        key_words['quantity_pairs'] = quantity
        key_words['range_quantity_pairs'] = range_quantity


        return key_words

    def manage_multiple_group(self,ent_groups,words):
        participants = ent_groups[0]['entities'].copy()
        start_span = ent_groups[0]['span']
        columns = []
        col_flag = False
        for g in ent_groups[1:]:
            # print(words[start_span[1]:g['span'][0]+1])
            if any([item in words[start_span[1]:g['span'][0]+1] for item in ['to','at','on','for']]):
                columns.extend(g['entities'])
                col_flag = True
            elif any([item in words[start_span[1]:g['span'][0]+1] for item in ['and','or','with']]):
                if col_flag==True:
                    columns.extend(g['entities'])
                else:
                    participants.extend(g['entities'])
            else:
                columns.extend(g['entities'])
                col_flag=True
            start_span = g['span']
        if len(columns)==0:
            columns = ent_groups[-1]['entities']
        return participants,columns

    def clean_ent(self,ents):
        no_use_words = ['—',':','.']+[' '+tmp+' ' for tmp in ['a','the','case','bonus','shelf','court']+list(self.numbers.keys()) + self.no_use_single_words]
        for eidx,ent in enumerate(ents):
            ents[eidx] = ' '+ents[eidx]
            for wd in no_use_words:
                ents[eidx] = ents[eidx].replace(wd,' ')
            ents[eidx] = ents[eidx].strip()
        return ents

    def extract_participants_columns(self,ent_groups,range_quantity,words,doc):

        columns = []
        participants = []
        modified_doc = doc

        if range_quantity:
            for rq in range_quantity:
                column,modified_doc = self.extract_columns_from_range(rq['text'],modified_doc)
                columns.extend(column)
            if len(ent_groups)>=2:
                participants, tmp_columns = self.manage_multiple_group(ent_groups, words)
                columns.extend(tmp_columns)
            elif len(ent_groups)==1:
                participants = ent_groups[0]['entities']
            else:
                participants = columns
        elif len(ent_groups) >= 2:
            participants, columns = self.manage_multiple_group(ent_groups,words)
        elif len(ent_groups) == 1:
            participants = ent_groups[0]['entities']
            columns = ent_groups[0]['entities']
        # print(participants)
        participants = self.clean_ent(participants)
        columns = self.clean_ent(columns)
        participants = set(participants)
        columns = set(columns)
        same = list(set.intersection(participants,columns))
        new_columns = [c for c in columns if (c not in same) ]
        participants = list(participants)
        if len(new_columns)==0:
            new_columns = columns

        # print('participants',participants,'\n')
        # print('columns',columns,'\n')
        # for idx,col in enumerate(columns):
        #     columns[idx] = list(set(col))
        return participants,new_columns,modified_doc

    def get_cp_ner_results(self,texts):
        tokens = self.predictior_cp.predict_batch_json(inputs=[{'sentence':text} for text in texts])
        dps = self.predictior_dp.predict_batch_json(inputs=[{'sentence':text} for text in texts])
        ent_res = []
        for text in texts:

            try:
                tmp_res = self.predictor_ner.predict(text)
                ent_res.append(tmp_res)
            except Exception as e:
                print(text)
                ent_res.append({'words':[],'tags':[]})
        # ent_res = self.predictor_ner.predict_batch_json(inputs=[{'sentence':text} for text in texts])
        # print(tokens[0].keys())
        for i in range(len(tokens)):
            del tokens[i]['class_probabilities']
        return tokens,ent_res,dps
    
    @classmethod
    def check_contain_upper(cls, password):
        pattern = re.compile('[A-Z]+')
        match = pattern.findall(password)
        if match:
            return True
        else:
            return False


    @classmethod
    def tokenize_sentence(self, sentence):
        tokens = nltk.word_tokenize(sentence)
        return tokens

    def normalize_tags(self, tagged):
        n_tagged = []
        for t in tagged:
            if t[1] == "NP-TL" or t[1] == "NP":
                n_tagged.append((t[0], "NNP"))
                continue
            if t[1].endswith("-TL"):
                n_tagged.append((t[0], t[1][:-3]))
                continue
            if t[1].endswith("S"):
                n_tagged.append((t[0], t[1][:-1]))
                continue
            n_tagged.append((t[0], t[1]))
        return n_tagged
        # Extract the main topics from the sentence



    @classmethod
    def get_NP(cls, tree, nps):
        if isinstance(tree, dict):
            if "children" not in tree:
                if tree['nodeType'] in ["NNP","NN"]:
                    # print(tree['word'])
                    # print(tree)
                    nps.append(tree['word'])
            elif "children" in tree:
                if tree['nodeType'] in ["NNP","NN"]:
                    # print(tree['word'])
                    nps.append(tree['word'])
                    cls.get_NP(tree['children'], nps)
                else:
                    cls.get_NP(tree['children'], nps)
        elif isinstance(tree, list):
            for sub_tree in tree:
                cls.get_NP(sub_tree, nps)

        return nps


    @classmethod
    def get_subjects(cls, tree):
        subject_words = []
        subjects = []
        for subtree in tree['children']:
            if subtree['nodeType'] == "VP" or subtree['nodeType'] == 'S' or subtree['nodeType'] == 'VBZ':
                subjects.append(' '.join(subject_words))
                subject_words.append(subtree['word'])
            else:
                subject_words.append(subtree['word'])
        return subjects



    @classmethod
    def search_entity_with_tags(cls, tags, words):
        if ('B-V' in tags):
            verb_idx = tags.index('B-V')
        else:
            return [], []
        subj, obj = [], []
        flag = False
        for idx in range(0, verb_idx):
            tag = tags[idx]
            if (tag != 'I-V'):
                if (tag.find('B-') != -1):
                    subj.append(words[idx])
                elif (tag.find('I-') != -1):
                    if (len(subj) != 0):
                        subj[-1] += ' %s' % words[idx]

        for idx in range(verb_idx + 1, len(tags)):
            tag = tags[idx]
            if (tag != 'I-V'):
                if (tag.find('B-') != -1):
                    obj.append(words[idx])
                elif (tag.find('I-') != -1):
                    if (len(obj) != 0):
                        obj[-1] += ' %s' % words[idx]

        return subj, obj

    def analyze_srl_result(self, srl_result):
        srls, words = srl_result['verbs'], srl_result['words']
        triples = []
        for srl in srls:
            verb, des, tags = srl['verb'], srl['description'], srl['tags']
            verb = verb
            subj, obj = self.search_entity_with_tags(tags, words)
            triples.append({'verb': verb, 'subject': subj, 'object': obj})
        return triples

    def found_openie_srl(self, texts):
        openie_results = self.openie_predictor.predict_batch_json(inputs=[{'sentence': text} for text in texts])
        srl_results = self.srl_predictor.predict_batch_json(inputs=[{'sentence': text} for text in texts])
        openie_triples = [self.analyze_srl_result(tmp) for tmp in openie_results]
        srl_triples = [self.analyze_srl_result(tmp) for tmp in srl_results]
        return openie_results, srl_results, openie_triples, srl_triples

    def infer_important_words(self, words, pos_tags):
        # obtain pos tag
        attn_words = []

        def hasNumbers(inputString):
            return bool(re.search(r'\d', inputString))

        for idx in range(len(pos_tags)):
            tag = pos_tags[idx]
            if (tag in self.pos_useful_tag):
                attn_words.append(words[idx])
            elif (hasNumbers(words[idx])):
                attn_words.append(words[idx])

        # if (len(src_loc) > 0 or len(dest_loc) > 0):
        #     print(
        #         'All location candidate is: {}\n The src_loc is: {} The dest_loc is: {}\n'.format(all_loc_cdd, src_loc,
        #                                                                                           dest_loc))

        return attn_words
    @classmethod
    def extract_entity_allennlp(cls, words, tags):
        all_ents = []
        all_ents_test = []
        flag = True
        # for i, tag in enumerate(tags):
        #     if(tag!='O'):
        #         all_ents_test.append(words[i])
        tmp = []

        for i, tag in enumerate(tags):
            flag = True if (cls.check_contain_upper(words[i])) else False
            if (tag != 'O' or flag):
                tmp.append(words[i])
            if (len(tmp) != 0 and ((tag == 'O' and flag == False) or i == (len(tags) - 1))):
                all_ents.append(' '.join(tmp))
                tmp = []

        # assert(' '.join(all_ents_test)==' '.join(all_ents)),'{}, {}, {}, {}'.format(all_ents,all_ents_test,words,tags)
        return all_ents

    @classmethod
    def judge_upper(self, text):
        bigchar = re.findall(r'[A-Z]', text)
        return (len(bigchar) > 0)





