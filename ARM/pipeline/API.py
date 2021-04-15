from rule_update_table import *
from rule_pattern import RulePattern
RP = RulePattern()

class API():
    def __init__(self):
        super(API,self).__init__()

        self.non_triggers = {}
        # self.non_triggers['greater'] = ['more', 'than', 'above', 'exceed', 'over','older','taller','bigger','larger','greater','higher']#,'RBR', 'JJR']
        # self.non_triggers['less'] = ['less', 'than', 'below', 'under','smaller']#,'RBR', 'JJR']

        # non_triggers['less'] = ['RBR', 'JJR', 'less', 'than', 'below', 'under','smaller']
        # self.non_triggers['after'] = ['after','followed']#non_triggers['before']
        self.non_triggers['to'] = [['lecture'],['given'],['lectures'],['to'],['from'],['on'],[['so'], ['does']]]
        self.non_triggers['different'] = ['different']
        self.non_triggers['last'] = [[['immediately'],['before','preceding']]]
        self.non_triggers['after_equal'] = [[['no'], ['earlier']]]
        self.non_triggers['adjacent'] = [['neighbouring']]
        self.non_triggers['before'] = ['before', 'above', 'precede', 'earlier']
        self.non_triggers['next'] = [[['next', 'adjacent'], ['to']], ['followed'],['follow'],[['immediately','directly'],['after']]]
        self.non_triggers['after'] = ['after'] + ['older','taller','bigger','larger','greater','higher']
        # self.non_triggers['must'] = ['must']
        self.non_triggers['same'] = [['same'],['also']]

        self.non_triggers['before_equal'] = [[['no'],['later']]]
        # self.non_triggers['or'] = ['or']
        # self.non_triggers['not'] = ['not', 'nor','no', 'never', "didn't", "won't", "wasn't", "isn't",
        #                           "haven't", "weren't", "won't", 'neither', 'none', 'unable',
        #                           'fail', 'different', 'outside', 'unable', 'fail','cannot','except']
        self.non_triggers['last_num'] = [[['one'],['of'],['the'],['last']]]
        # print(self.non_triggers['last_num'])
        self.non_triggers['first_num'] = [[['one'], ['of'], ['the'], ['first']]]
        for k,v in self.non_triggers.items():
            if not isinstance(v[0],list):
                self.non_triggers[k] = [[t] for t in v]
        self.APIs = {}
        self.APIs['greater'] = {
            'function': None,
            'tostr': lambda arg1, arg2: 'greater({}, {})'.format(arg1, arg2)
        }
        self.APIs['less'] = {
            'function': None,
            'tostr': lambda arg1, arg2: 'less({}, {})'.format(arg1, arg2)
        }
        self.APIs['before'] = {
            'function': None,
            'tostr': lambda arg1, arg2: 'before({}, {})'.format(arg1, arg2)
        }
        self.APIs['after'] = {
            'function': None,
            'tostr': lambda arg1, arg2: 'after({}, {})'.format(arg1, arg2)
        }
        self.APIs['to'] = {
            'function': None,
            'tostr': lambda par, col: 'assign_to({}, {})'.format(par, col)
        }
        self.APIs['must'] = {
            'function': None,
            'tostr': lambda ent, text: 'must({}, {})'.format(ent, text)
        }

        self.APIs['same'] = {
            'function': None,
            'tostr': lambda ent1, ent2: 'same({}, {})'.format(ent1, ent2)
        }

        self.APIs['next'] = {
            'function': None,
            'tostr': lambda ent1, ent2: 'next_to({}, {})'.format(ent1, ent2)
        }
    def neighbour_par_col(self,par_occur,col_occur,position):
        for idx in range(len(par_occur)):
            par_occur[idx]['distance'] = -(min(position) - par_occur[idx]['position'][1]) if par_occur[idx]['position'][1] >= min(position) else \
                par_occur[idx]['position'][0] - max(position)

        for idx in range(len(col_occur)):
            col_occur[idx]['distance'] = -(min(position) - col_occur[idx]['position'][1]) if col_occur[idx]['position'][
                                                                                              1] >= min(position) else \
                col_occur[idx]['position'][0] - max(position)

        all_ent = par_occur + col_occur
        sorted_par = sorted(par_occur, key=lambda k: abs(k['distance']))
        sorted_col = sorted(col_occur, key=lambda k: abs(k['distance']))
        sorted_ent = sorted(all_ent, key=lambda k: abs(k['distance']))
        # print('sorted entities is',sorted_ent)
        # print(sorted_par)
        # print(sorted_col)
        return sorted_par,sorted_col,sorted_ent


    def select_arguments(self,sorted_par, sorted_col, sorted_ent, position, tokens, mode):
        arg1, arg2 = None,None
        if mode == 'pfp':# participant, function_name, participant
            for tmp in sorted_par:
                if tmp['distance']<0 and arg1==None:
                    arg1 = tmp#['participant']
                elif tmp['distance']>0 and arg2==None:
                    arg2 = tmp#['participant']
            return arg1,arg2
        elif mode=='pfc/cfp':
            for tmp in sorted_ent:
                if tmp['distance'] < 0 and not arg1 and ((arg2 and arg2['type']!=tmp['type']) or not arg2):#and not arg1:
                    arg1 = tmp
                if tmp['distance'] > 0 and not arg2 and ((arg1 and arg1['type']!=tmp['type']) or not arg1):
                    arg2 = tmp
            return arg1, arg2
        elif mode=='cfc/pfp':
            for tmp in sorted_par:
                if tmp['distance'] < 0 and arg1 == None:
                    arg1 = tmp  # ['participant']
                elif tmp['distance'] > 0 and arg2 == None:
                    arg2 = tmp  # ['participant']
            tmp_arg1, tmp_arg2 = None, None
            for tmp in sorted_col:
                if tmp['distance'] < 0 and tmp_arg1 == None:
                    tmp_arg1 = tmp  # ['participant']
                elif tmp['distance'] > 0 and tmp_arg2 == None:
                    tmp_arg2 = tmp  # ['participant']

            if (arg1 and arg2) and not (tmp_arg1 and tmp_arg2):
                return arg1,arg2
            elif (tmp_arg1 and tmp_arg2) and not (arg1 and arg2):
                return tmp_arg1,tmp_arg2
            elif arg1 and arg2 and tmp_arg1 and tmp_arg2:
                if abs(tmp_arg1['distance'])+abs(tmp_arg2['distance']) < abs(arg1['distance'])+abs(arg2['distance']):
                    return arg1, arg2
                else:
                    return tmp_arg1, tmp_arg2
            else:
                return None,None

        elif mode== 'pfc':#participant, function, column
            for tmp in sorted_par:
                if tmp['distance']<0 and arg1==None:
                    arg1 = tmp#['participant']
            for tmp in sorted_col:
                if tmp['distance']>0 and arg2==None:
                    arg2 = tmp#['participant']
            return arg1,arg2
        elif mode=='eft':# entity (participant/column), function, sub-text
            for tmp in sorted_ent:
                if tmp['distance']>0:
                    arg1 = tmp['participant']
                    break
            pos = tokens[max(position):].find(',')
            pos_2 = tokens[max(position):].find('.')
            pos = min(pos,pos_2)
            sub_text = tokens[max(position):pos+1]
            return arg1, ' '.join(sub_text)
        elif mode=='efe':#entity, function, entity
            # print(sorted_ent)
            for tmp in sorted_ent:
                if tmp['distance']<0 and not arg1:
                    arg1 = tmp#['participant']
                if tmp['distance']>0 and not arg2:
                    arg2 = tmp#['participant']
            return arg1,arg2
        elif mode=='pp/cc':
            leading_type = None
            for tmp in sorted_ent:
                if not leading_type:
                    arg1 = tmp#['participant']
                    leading_type = tmp['type']
                elif leading_type and tmp['type']==leading_type:
                   arg2 = tmp
            return arg1,arg2

        elif mode =='common':
            for tmp in sorted_ent:
                arg1 = tmp if arg1==None else arg1
                arg2 = tmp if (arg1!=None and arg2==None) else arg2
            return arg1,arg2

    def select_arg_pairs(self,sorted_par,sorted_col,sorted_ent, position, tokens, mode='pfc/cfp'):
        arg_pairs = []
        cand_arg1, cand_arg2 = [], []
        neigbor_end_pos = [0,len(tokens)]
        stop_pos = [0,len(tokens)]+[i for i,x in enumerate(tokens) if x==','] + [i for i,x in enumerate(tokens) if x=='.']
        ranked_stop_pos = sorted(stop_pos)
        for pid,pos in enumerate(ranked_stop_pos[:-1]):
            if pos < min(position) and ranked_stop_pos[pid+1] > max(position):
                neigbor_end_pos[0]=pos
                neigbor_end_pos[1]=ranked_stop_pos[pid+1]+1
                break
        # print(tokens)
        # print(neigbor_end_pos)
        if mode == 'pfc/cfp':
            for tmp in sorted_ent:
                if tmp['distance'] < 0 and tmp['position'][1]<=neigbor_end_pos[1] and tmp['position'][0]>=neigbor_end_pos[0] :
                    cand_arg1.append(tmp)
                if tmp['distance']>0 and tmp['position'][1]<=neigbor_end_pos[1] and tmp['position'][0]>=neigbor_end_pos[0]:
                    cand_arg2.append(tmp)
            for arg1 in cand_arg1:
                for arg2 in cand_arg2:
                    if arg1['type']!=arg2['type']:
                        arg_pairs.append((arg1,arg2))
        elif mode=='pfp/cfc':
            for tmp in sorted_ent:
                if tmp['distance'] < 0 and tmp['position'][1]<=neigbor_end_pos[1] and tmp['position'][0]>=neigbor_end_pos[0] :
                    cand_arg1.append(tmp)
                if tmp['distance']>0 and tmp['position'][1]<=neigbor_end_pos[1] and tmp['position'][0]>=neigbor_end_pos[0]:
                    cand_arg2.append(tmp)
            for arg1 in cand_arg1:
                for arg2 in cand_arg2:
                    if arg1['type']==arg2['type'] and arg1['participant']!=arg2['participant']:
                        arg_pairs.append((arg1,arg2))
        #fee case
        if not arg_pairs:
            for tmp in sorted_ent:
                if tmp['position'][1]<neigbor_end_pos[1] and tmp['position'][0]>neigbor_end_pos[0] :
                    cand_arg1.append(tmp)
            for arg1 in cand_arg1:
                for arg2 in cand_arg1:
                    if arg1['type'] != arg2['type']:
                        r = arg1 if arg1['type']=='row' else arg2
                        c = arg2 if arg2['type']=='column' else arg1
                        if ((r,c) not in arg_pairs) and arg1['participant']!=arg2['participant']:
                            arg_pairs.append((r,c))
        for i, pair in enumerate(arg_pairs):
            # print(pair)
            start_pos = min(list(pair[0]['position'])+list(pair[1]['position']))
            end_pos = max(list(pair[0]['position'])+list(pair[1]['position']))
            if ',' in tokens[start_pos:end_pos+1]:
                arg_pairs.pop(i)
        return arg_pairs


    def func_to(self, par_occur, col_occur, tokens, position, table, contain_negation,all_rows,all_columns):
        sorted_par, sorted_col, sorted_ent = self.neighbour_par_col(par_occur, col_occur, position)
        # arg1, arg2 = self.select_arguments(sorted_par, sorted_col, sorted_ent, position, tokens, 'pfc/cfp')
        arg_pairs = self.select_arg_pairs(sorted_par,sorted_col,sorted_ent, position, tokens,'pfc/cfp')
        # print('argument pairs',arg_pairs)
        return_func = []

        for pair in arg_pairs:
            rule1 = TO(pair[0], pair[1], contain_negation, all_rows, all_columns)
            # print(rule1)

            return_func.append(rule1)
        return return_func
        # if arg1 and arg2:
        #     row_idx = arg1['idx'] if arg1['type'] == 'row' else arg2['idx']
        #     col_idx = arg1['idx'] if arg1['type'] == 'column' else arg2['idx']
        #     v = False if contain_negation else True
        #     table[row_idx][col_idx] = v
        # return table, {'function': 'to', 'arguments': (arg1, arg2)}  # self.APIs['to']['tostr'](arg1, arg2)

    def func_next(self,par_occur,col_occur, tokens, position, table, contain_negation, all_rows, all_columns):
        sorted_par, sorted_col, sorted_ent = self.neighbour_par_col(par_occur, col_occur, position)
        arg1, arg2 = self.select_arguments(sorted_par, sorted_col, sorted_ent, position, tokens, 'cfc/pfp')
        if arg1 and arg2:
            rule = NEXT(arg1,arg2, contain_negation,all_rows,all_columns)
            return [rule]
        else:
            return []

    def func_adjacent(self,par_occur,col_occur, tokens, position, table, contain_negation, all_rows, all_columns):
        sorted_par, sorted_col, sorted_ent = self.neighbour_par_col(par_occur, col_occur, position)
        arg1, arg2 = self.select_arguments(sorted_par, sorted_col, sorted_ent, position, tokens, 'cfc/pfp')
        if arg1 and arg2:
            rule = ADJACENT(arg1,arg2, contain_negation,all_rows,all_columns)
            return [rule]
        else:
            return []

    def func_last(self,par_occur,col_occur, tokens, position, table, contain_negation, all_rows, all_columns):
        sorted_par, sorted_col, sorted_ent = self.neighbour_par_col(par_occur, col_occur, position)
        arg1, arg2 = self.select_arguments(sorted_par, sorted_col, sorted_ent, position, tokens, 'cfc/pfp')
        if arg1 and arg2:
            rule = LAST(arg1,arg2, contain_negation,all_rows,all_columns)
            return [rule]
        else:
            return []

    def func_after(self,par_occur,col_occur, tokens, position, table, contain_negation, all_rows, all_columns):
        sorted_par, sorted_col, sorted_ent = self.neighbour_par_col(par_occur, col_occur, position)
        arg_pairs = self.select_arg_pairs(sorted_par, sorted_col, sorted_ent, position, tokens, 'pfp/cfc')
        # print('argument pairs',arg_pairs)
        return_func = []

        for pair in arg_pairs:
            rule1 = AFTER(pair[0], pair[1], contain_negation, all_rows, all_columns)
            # print(rule1)

            return_func.append(rule1)
        return return_func
    def func_before_equal(self, par_occur, col_occur, tokens, position, table, contain_negation, all_rows, all_columns):
        sorted_par, sorted_col, sorted_ent = self.neighbour_par_col(par_occur, col_occur, position)
        arg_pairs = self.select_arg_pairs(sorted_par, sorted_col, sorted_ent, position, tokens, 'pfp/cfc')
        # print('argument pairs',arg_pairs)
        return_func = []
        for pair in arg_pairs:
            rule1 = BeforeEqual(pair[0], pair[1], contain_negation, all_rows, all_columns)
            # print(rule1)

            return_func.append(rule1)
        return return_func

    def func_after_equal(self, par_occur, col_occur, tokens, position, table, contain_negation, all_rows,
                          all_columns):
        sorted_par, sorted_col, sorted_ent = self.neighbour_par_col(par_occur, col_occur, position)
        arg_pairs = self.select_arg_pairs(sorted_par, sorted_col, sorted_ent, position, tokens, 'pfp/cfc')
        # print('argument pairs',arg_pairs)
        return_func = []
        for pair in arg_pairs:
            rule1 = BeforeEqual(pair[1], pair[0], contain_negation, all_rows, all_columns)
            # print(rule1)

            return_func.append(rule1)
        return return_func

        # arg1, arg2 = self.select_arguments(sorted_par, sorted_col, sorted_ent, position, tokens, 'cfc/pfp')
        # if arg1 and arg2:
        #     rule = AFTER(arg1,arg2, contain_negation,all_rows,all_columns)
        #     return [rule]
        # else:
        #     return []
    def func_before(self, par_occur, col_occur, tokens, position, table, contain_negation, all_rows, all_columns):
        sorted_par, sorted_col, sorted_ent = self.neighbour_par_col(par_occur, col_occur, position)
        arg_pairs = self.select_arg_pairs(sorted_par, sorted_col, sorted_ent, position, tokens, 'pfp/cfc')
        # print('argument pairs',arg_pairs)
        return_func = []

        for pair in arg_pairs:
            rule1 = BEFORE(pair[0], pair[1], contain_negation, all_rows, all_columns)
            # print(rule1)

            return_func.append(rule1)
        return return_func

        # arg1, arg2 = self.select_arguments(sorted_par, sorted_col, sorted_ent, position, tokens, 'cfc/pfp')
        # if arg1 and arg2:
        #     rule = AFTER(arg1,arg2, contain_negation,all_rows,all_columns)
        #     return [rule]
        # else:
        #     return []
    def func_different(self, par_occur, col_occur, tokens, position, table, contain_negation,all_rows,all_columns):
        sorted_par, sorted_col, sorted_ent = self.neighbour_par_col(par_occur, col_occur, position)
        arg1, arg2 = self.select_arguments(sorted_par, sorted_col, sorted_ent, position, tokens, 'pp/cc')
        if arg1 and arg2:
            rule = DIFFERENT(arg1,arg2,contain_negation,all_rows,all_columns)
            return [rule]
        else:
            return []

    def func_same(self, par_occur, col_occur, tokens, position, table, contain_negation,all_rows,all_columns):
        sorted_par, sorted_col, sorted_ent = self.neighbour_par_col(par_occur, col_occur, position)
        arg1, arg2 = self.select_arguments(sorted_par, sorted_col, sorted_ent, position, tokens, 'pp/cc')
        if arg1 and arg2:
            rule = SAME(arg1, arg2, contain_negation, all_rows, all_columns)
            return [rule]
        else:
            return []

    def func_if_then(self,if_func,then_func,all_rows,all_columns):
        ifthen_rule = IFTHEN(if_func,then_func,all_rows,all_columns)
        return [ifthen_rule]

    def func_last_num(self, par_occur, col_occur, tokens, position, table, contain_negation,all_rows,all_columns):
        sorted_par, sorted_col, sorted_ent = self.neighbour_par_col(par_occur, col_occur, position)
        if max(position)!=len(tokens)-1:
            num = tokens[max(position)+1]
        else:
            return []
        if num in RP.numbers.keys():
            num = RP.numbers[num]

            # print(num,tokens,position)
            close_ent = None
            for ent in sorted_ent:
                if ent['distance']<0 and ent['type']=='row':
                    close_ent = ent
                    break
            if close_ent and num:
                rule = LastNum(close_ent,num,all_rows,all_columns)
                return [rule]
            else:
                return []
        else:
            return self.func_to(par_occur, col_occur, tokens, position, table, contain_negation, all_rows, all_columns)

    def func_first_num(self, par_occur, col_occur, tokens, position, table, contain_negation,all_rows,all_columns):
        sorted_par, sorted_col, sorted_ent = self.neighbour_par_col(par_occur, col_occur, position)
        if max(position) != len(tokens) - 1:
            num = tokens[max(position) + 1]
        else:
            return []
        if num in RP.numbers.keys():
            num = RP.numbers[num]
            close_ent = None
            for ent in sorted_ent:
                if ent['distance']<0 and ent['type']=='row':
                    close_ent = ent
                    break
            if close_ent and num:
                # print(num)
                rule = FirstNum(close_ent,num,all_rows,all_columns)
                return [rule]
            else:
                return []
        else:
            return self.func_to(par_occur, col_occur, tokens, position, table, contain_negation,all_rows,all_columns)



#non_triggers['istype_s_n'] = ['is', 'are', 'were', 'was', 'be', 'within', 'one', 'of']
#non_triggers['istype_n_s'] = ['is', 'are', 'were', 'was', 'be', 'within', 'one', 'of']
#non_triggers['count'] = ['there', 'num', 'amount', 'have', 'has', 'had', 'are', 'more']
#non_triggers['max'] = [k for k, v in triggers.iteritems() if v == 'max']
#non_triggers['argmax'] = [k for k, v in triggers.iteritems() if v == 'argmax']
#non_triggers['and'] = ['and', 'while', 'when', ',', 'neither', 'none', 'all', 'both']
#non_triggers['neither'] = ['neither', 'none', 'not', "'nt", 'both']
