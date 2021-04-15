#https://pypi.org/project/anytree/
from anytree import Node, RenderTree, NodeMixin,LevelOrderIter
import json
import copy
import itertools

def merge_option_functions(option_functions):
    for id,funcs in enumerate(option_functions):
        args = []
        for j,func in enumerate(funcs):
            if not isinstance(func,TO):
                args.append(func.arg1)
                args.append(func.arg2)
        for j, func in enumerate(funcs):
            if isinstance(func,TO):
                if func.arg1 in args or func.arg2 in args:
                    option_functions[id].pop(j)
    return option_functions

def filter_func(kw,sorted_funcs, positions,tokens):
    # In the case: Trapezoid must either be earlier than both Reciprocity and Salammbo or later than both Reciprocity and Salammbo.
    # for previous function, if argument go across a or and multiple functions is not allowed
    # for the last function, if argument all exist and is not the subj of the previous function is not allowed
    prev_subj = []
    prev_obj = []
    prev_so_pair = []
    prev_obj_pos = []
    new_funcs = []
    all_func_type = []
    for i, func in enumerate(sorted_funcs):
        if (type(func), positions[i]) not in all_func_type:
            all_func_type.append((type(func), positions[i]))
    for type_idx, func_type in enumerate(all_func_type):
        for i, func in enumerate(sorted_funcs):
            if (type(func), positions[i]) != func_type:
                continue
            flag = True
            arg1_pos = func.arg1['position']
            arg2_pos = func.arg2['position']
            end_pos = max(list(arg1_pos) + list(arg2_pos))
            if type_idx != len(all_func_type) - 1:
                if any([isinstance(func, tmp) for tmp in [BEFORE, LAST]]):
                    prev_subj.append(func.arg2['participant'])
                    prev_obj.append(func.arg1['participant'])
                    # prev_so_pair.append((func.arg2['participant'],func.arg1['participant']))
                    prev_obj_pos.append(max(func.arg1['position']))
                else:
                    prev_subj.append(func.arg1['participant'])
                    prev_obj.append(func.arg2['participant'])
                    # prev_so_pair.append((func.arg1['participant'], func.arg2['participant']))
                    prev_obj_pos.append(func.arg2['position'])
                for j, tmp_func in enumerate(all_func_type[type_idx+1:]):
                    if (end_pos > max(list(all_func_type[type_idx + j + 1][1]))) \
                            and (kw in tokens[max(list(positions[i])): max(list(all_func_type[type_idx + j + 1][1]))]) \
                            and any(
                        [item in range(max(list(positions[i])), max(list(all_func_type[type_idx + j + 1][1]))) for item
                         in prev_obj_pos]):
                        flag = False
                        break
            else:
                if any([isinstance(func, tmp) for tmp in [BEFORE, LAST]]):
                    subj, obj = func.arg2['participant'], func.arg1['participant']
                else:
                    subj, obj = func.arg1['participant'], func.arg2['participant']
                if (obj in prev_subj + prev_obj and subj in prev_obj) :
                    flag = False
            if flag:
                new_funcs.append(sorted_funcs[i])
    # print(new_funcs)
    return new_funcs

def merge_funcs_with_and(funcs,tokens,positions):
    if not ('and' in tokens):
        return funcs
    if len(funcs)<2:
        return funcs
    # print(funcs)
    def filter_funcs(sorted_funcs, sorted_positions):
        drop = []
        prev_subj, prev_obj = [],[]
        all_func_type = []
        for i, func in enumerate(sorted_funcs):
            if (type(func), sorted_positions[i]) not in all_func_type:
                all_func_type.append((type(func), sorted_positions[i]))
        for type_idx, func_type in enumerate(all_func_type):
            for i, func in enumerate(sorted_funcs):
                flag = True
                if (type(func), sorted_positions[i]) != func_type:
                    continue
                max_pos = max(list(func.arg1['position'])+list(func.arg2['position']))
                if type_idx!=len(all_func_type)-1:
                    next_pos = min(all_func_type[type_idx+1][1])
                    if max_pos >= next_pos:
                        drop.append(i)
                        flag = False
                if any([isinstance(func, tmp) for tmp in [BEFORE, LAST]]):
                    subj, obj = func.arg2['participant'], func.arg1['participant']
                else:
                    subj, obj = func.arg1['participant'], func.arg2['participant']
                if subj in prev_obj:
                    drop.append(i)
                    flag = False
                if flag:
                    prev_subj.append(subj)
                    prev_obj.append(obj)
            # if any([isinstance(sorted_funcs[-1], tmp) for tmp in [BEFORE, LAST]]):
            #     subj, obj = sorted_funcs[-1].arg2['participant'], sorted_funcs[-1].arg1['participant']
            # else:
            #     subj, obj = sorted_funcs[-1].arg1['participant'], sorted_funcs[-1].arg2['participant']
            # if subj in prev_obj:
            #     drop.append(len(sorted_funcs)-1)
        output_funcs = []
        for i,func in enumerate(sorted_funcs):
            if i not in drop:
                output_funcs.append(func)
        return output_funcs

    and_pos = [idx for idx in range(len(tokens)) if tokens[idx] == 'and']
    # no_process_funcs = [fun for fun in funcs if (isinstance(fun,OR) or isinstance(fun,IFTHEN))]
    # rest_funcs = [fun for fun in funcs if fun not in no_process_funcs]
    positions = [(i, positions[i]) for i in range(len(positions))]
    sorted_positions = sorted(positions, key=lambda k: max(k[1]))
    sorted_funcs = [funcs[tmp[0]] for tmp in sorted_positions]
    rest_func = []
    for i in range(len(sorted_funcs)):
        if any([isinstance(sorted_funcs[i], tmp) for tmp in [OR, IFTHEN, IFF,AND,UNLESS]]):
            rest_func.append(sorted_funcs[i])
            sorted_funcs.pop(i)
            sorted_positions.pop(i)
    # sorted_funcs = sorted(rest_funcs,key=lambda k:max(k.arg1['position']))
    sorted_positions = [pos[1] for pos in sorted_positions]

    sorted_funcs = filter_funcs(sorted_funcs, sorted_positions)
    prev_rule_set, after_rule_set = [], []
    new_funcs = []
    for pos in and_pos:
        for i, func in enumerate(sorted_funcs):
            all_arg = [func.arg1, func.arg2]
            end_pos = max([max(arg['position']) for arg in all_arg])
            if end_pos > pos:
                after_rule_set.append(func)
            else:
                prev_rule_set.append(func)
        # print(prev_rule_set,after_rule_set)
        new_funcs.append(
            AND(copy.deepcopy(prev_rule_set), copy.deepcopy(after_rule_set), func.participants_name, func.column_names))

    # print(new_funcs + rest_func)
    return new_funcs + rest_func
def merge_funcs_with_neither(funcs,tokens,positions):
    if ('neither' in tokens or 'Neither' in tokens) and 'nor' in tokens:
        all_funcs =[]
        neither_func, nor_func = [],[]
        if 'neither' in tokens:
            neither_pos = tokens.index('neither')
        if 'Neither' in tokens:
            neither_pos = tokens.index('Neither')
        nor_pos = tokens.index('nor')
        neither_part = range(neither_pos,nor_pos+1)
        nor_part = range(nor_pos,len(tokens))
        for i,func in enumerate(funcs):
            min_pos = min(list(positions[i]))
            max_pos = max(list(positions[i]))
            if min_pos in neither_part and max_pos in neither_part:
                neither_func.append(func)
            elif min_pos in nor_part and max_pos in nor_part:
                nor_func.append(func)
        if funcs:
            all_funcs.append(NEITHER(neither_func,nor_func,func.participants_name,func.column_names))
        return all_funcs
    else:
        return funcs
def merge_funcs_with_unless(funcs,tokens,positions):
    if any([item in ['Unless','unless'] for item in tokens]):
        all_funcs = []
        unless_func,rest_func = [],[]
        if ',' in tokens:
            comma_pos = tokens.index(',')
        else:
            comma_pos = len(tokens)-1
        if 'Unless' in tokens:
            unless_pos = tokens.index('Unless')
            unless_part = (unless_pos,comma_pos+1)
            rest_part = (comma_pos,len(tokens))
        elif 'unless' in tokens:
            unless_pos = tokens.index('unless')
            unless_part = (0,unless_pos)
            rest_part = (comma_pos,len(tokens))
        for i,func in enumerate(funcs):
            min_pos = min(list(positions[i])+list(func.arg2['position'])+list(func.arg1['position']))
            max_pos = max(list(positions[i])+list(func.arg2['position'])+list(func.arg1['position']))
            if min_pos in range(unless_part[0],unless_part[1]) and max_pos in range(unless_part[0],unless_part[1]):
                unless_func.append(func)
            elif min_pos in range(rest_part[0],rest_part[1]) and max_pos in range(rest_part[0],rest_part[1]):
                rest_func.append(func)
        if funcs:
            all_funcs.append(UNLESS(unless_func,rest_func,func.participants_name,func.column_names))
        return all_funcs
    else:
        return funcs

def merge_funcs_with_iff(funcs,tokens,ifpos,thenpos,positions):

    new_func = []
    p_part = list(range(0,ifpos+1))#tokens[:ifpos+1]
    q_part = list(range(thenpos,len(tokens)))#tokens[thenpos:]
    p_func,q_func = [],[]
    sorted_funcs = sorted(funcs, key=lambda k: max(k.arg1['position']))
    pq_flag = [True for i in range(len(funcs))]
    for i,func in enumerate(sorted_funcs):
        pos = positions[i]
        if (min(pos) in p_part or max(pos) in p_part) and (min(func.arg1['position']+func.arg2['position']) in p_part and max(func.arg1['position']+func.arg2['position']) in p_part):
            p_func.append(func)
        elif (min(pos) in q_part or max(pos) in q_part) and (min(func.arg1['position']+func.arg2['position']) in q_part and max(func.arg1['position']+func.arg2['position']) in q_part):
            q_func.append(func)
    if sorted_funcs:
        new_func.append(IFF(p_func,q_func,sorted_funcs[0].participants_name,sorted_funcs[0].column_names))
    return new_func


def merge_funcs_with_or(funcs,tokens,positions):
    if not ('or' in tokens):
        return funcs
    new_funcs = []
    if len(funcs)<2:
        return funcs
    # print(funcs)

    or_pos = [idx for idx in range(len(tokens)) if tokens[idx]=='or']
    # no_process_funcs = [fun for fun in funcs if (isinstance(fun,OR) or isinstance(fun,IFTHEN))]
    # rest_funcs = [fun for fun in funcs if fun not in no_process_funcs]
    positions = [(i,positions[i]) for i in range(len(positions))]
    sorted_positions = sorted(positions,key=lambda k: max(k[1]))
    sorted_funcs = [funcs[tmp[0]] for tmp in sorted_positions]
    rest_func = []
    for i in range(len(sorted_funcs)):
        if any([isinstance(sorted_funcs[i],tmp) for tmp in [OR,IFTHEN,IFF]]):
            rest_func.append(sorted_funcs[i])
            sorted_funcs.pop(i)
            sorted_positions.pop(i)
    # sorted_funcs = sorted(rest_funcs,key=lambda k:max(k.arg1['position']))
    sorted_positions = [pos[1] for pos in sorted_positions]
    sorted_funcs = filter_func('or',sorted_funcs,sorted_positions,tokens)
    prev_rule_set, after_rule_set = [], []
    new_funcs = []
    for pos in or_pos:
        for i, func in enumerate(sorted_funcs):
            all_arg = [func.arg1, func.arg2]
            end_pos = max([max(arg['position']) for arg in all_arg])
            if end_pos > pos:
                after_rule_set.append(func)
            else:
                prev_rule_set.append(func)
        if (len(after_rule_set)==2 and not prev_rule_set) or (len(prev_rule_set)==2 and not after_rule_set):
            if after_rule_set:
                prev_rule_set = [after_rule_set[0]]
                after_rule_set.pop(0)
            elif prev_rule_set:
                after_rule_set = [prev_rule_set[0]]
                prev_rule_set.pop(0)
        # print(prev_rule_set,after_rule_set)
        new_funcs.append(OR(copy.deepcopy(prev_rule_set), copy.deepcopy(after_rule_set), func.participants_name, func.column_names))
    return new_funcs + rest_func



class Node(NodeMixin):
    def __init__(self, name, assignments, ents, parent=None, children=None):
        super(Node, self).__init__()
        self.name = name
        self.assignment = copy.deepcopy(assignments)

        self.ents = ents
        self.parent = parent
        if children:
            self.children = children
    def __repr__(self):
        return '{}'.format(list(set(self.assignment)))

class RuleTree():
    def __init__(self,rows,columns,rules,old_assignments,question_type):
        super(RuleTree, self).__init__()
        self.original_assign = old_assignments
        self.rows = rows
        self.columns = columns
        self.rules = rules
        self.root = None
        self.question_type=question_type

    def obtain_value(self,assignments):
        ents = [set() for i in range(len(assignments))]
        for i,a in enumerate(assignments):
            for item in a:
                ents[i].add(item[0])
                ents[i].add(item[1])

        return assignments,list(ents)

    def obtain_root_and_construct_tree(self):
        begin_status, ents = self.obtain_value(self.original_assign)
        self.root = Node('root', [], [])
        children = []
        # print(begin_status)
        for id,(status, ent) in enumerate(zip(begin_status,ents)):
            child = Node(' '.join(ent)+str(id), list(set(status)),
                         list(set(ent)),parent=self.root)
            children.append(child)
        rule_id = 0
        if children:
            for subroot in children:
                self.construct_sub_tree(subroot,rule_id)
        else:
            subroot = Node(' ', [],
                         [],parent=self.root)
            self.construct_sub_tree(subroot, rule_id)
        # print('total number of leaf nodes {}'.format(len(self.obtain_leaf_nodes())))
        # print(RenderTree(self.root))
        final_leaf = []
        each_level_node = [[] for i in range(len(self.rules)+1)]
        # print(RenderTree(self.root))
        for node in LevelOrderIter(self.root):
            each_level_node[node.depth-1].append(node)
            if node.depth == len(self.rules)+1:
                final_leaf.append(node)

        return final_leaf,each_level_node
        # print(self.obtain_leaf_nodes())

    def construct_sub_tree(self,root,rule_id):
        if rule_id==len(self.rules):
            return
        rule = self.rules[rule_id]
        # print(rule)
        assignment, new_ents = rule.find_assignment(root.assignment, root.ents, self.question_type)
        # if conflict:
        #     del root
        #     return output
        rule_id = rule_id + 1
        childrens = self.update_nodes(root,assignment, new_ents)
        for child in childrens:
            self.construct_sub_tree(child,rule_id)


    def update_nodes(self,root,new_assignment, new_ents):
        previous_assignment = root.assignment
        children = []
        for id,assign in enumerate(new_assignment):
            child = Node(' '.join(root.ents+new_ents)+str(id), list(set(previous_assignment+assign)),
                         list(set(root.ents+new_ents)),parent=root)
            children.append(child)
        return children

    def obtain_leaf_nodes(self):
        return self.root.leaves

class Rule():
    def __init__(self,arg1,arg2,all_pars,all_cols):
        super(Rule, self).__init__()
        self.conditions = []
        self.results = []
        self.participants = []
        self.columns = []
        self.participants_name = all_pars
        self.column_names = all_cols
        self.arg1 = arg1 if arg1 else None
        self.arg2 = arg2 if arg2 else None
        # self.position = position


    def __repr__(self):
        output = self.__class__.__name__+ ':\n'
        if self.arg1 and self.arg2:
            output_dict = {'arguments':[self.arg1['participant'],self.arg2['participant']]}
        else:
            output_dict = {'arguments': [self.arg1, self.arg2]}

        output += json.dumps(output_dict,indent=4)+'\n'
        return output
        # return str({
        #     'participants':[self.participants_name[i] for i in self.participants],
        #     'columns':[self.column_names[i] for i in self.columns]
        # })
    def find_ent_related(self,ent,old_assignment):
        ent_assign = []
        non_empty  =[]
        assign_flag = False
        for assign in old_assignment:
            if assign[2]:
                non_empty.append(assign[1])
            if ent in assign:
                ent_assign.append(assign)
                if assign[2]:
                    assign_flag = True#assign flag is used to record whether this entities has been assigned to some space

        return ent_assign,assign_flag,non_empty
    def generate_new_assign_by_limitation(self,ent,ent_assign,non_empty,old_assignment):
        news = []
        cant_exist = [item[1] for item in ent_assign if not item[2]] + non_empty
        rest_col = [item for item in self.column_names if item not in cant_exist]
        for col in rest_col:
            for assign in old_assignment:
                news.append(assign+[(ent,col,True)])
                for col2 in self.column_names:
                    if col2!=col:
                        news[-1].append((ent,col,False))
        return news
    def find_assignment(self,old_assignment, old_ents,question_type):
        new_pars = []
        new_cols = []
        now_assignments=[]
        for ent in self.participants:
            if ent not in old_ents:
                new_pars.append(ent)
            else:
                ent_assign, assign_flag, non_empty = self.find_ent_related(ent, old_assignment)
                if not now_assignments and not assign_flag:
                    now_assignments = self.generate_new_assign_by_limitation(ent, ent_assign, non_empty, [old_assignment])
                elif now_assignments and not assign_flag:
                    now_assignments = self.generate_new_assign_by_limitation(ent, ent_assign, non_empty,
                                                                             now_assignments)

        for ent in self.columns:
            if ent not in old_ents:
                new_cols.append(ent)
        new_pars = list(set(new_pars))
        new_cols = list(set(new_cols))
        if not now_assignments:
            new_all_possible_assign = self.generate_possible_assignment(new_pars,[old_assignment],question_type)
        else:
            new_all_possible_assign = self.generate_possible_assignment(new_pars, now_assignments,question_type)
        # print('possible assigment: ',new_all_possible_assign)
        satisfied_assign = self.find_satisfied_assignment(new_all_possible_assign)
        # print('satisfy',satisfied_assign)
        return satisfied_assign,new_pars+new_cols

    def generate_possible_assignment(self,new_pars,old_assginment,problem_type='ordering'):
        '''
        :param new_pars:
        :param occur: 'once': each participant can occur only in one position, 'multiple': each participant can occur in multiple position
        :return: all possible value of participant regardless of rule
        '''
        all_possible = []
        cache = []
        if not new_pars:
            return old_assginment
        else:
            if problem_type == 'grouping':
                for par in new_pars:
                    # print(cache)
                    tmp = []
                    for col1 in self.column_names:
                        if cache:
                            for item in cache:
                                tmp_item = copy.deepcopy(item)
                                for col2 in self.column_names:
                                    if col1!=col2:
                                        tmp_item.append((par, col2, False))
                                    else:
                                        tmp_item.append((par,  col2,  True))
                                tmp.append(tmp_item)
                        else:
                            one = []
                            for col2 in self.column_names:
                                if col1 != col2:
                                    one.append((par, col2, False))
                                else:
                                    one.append((par, col2, True))
                            tmp.append(one)
                    cache = tmp
                for item in cache:
                    for assign in old_assginment:
                        all_possible.append(assign + item)
            elif problem_type == 'ordering':
                for vv in old_assginment:
                    cache = []
                    exist_col = []
                    for v in vv:
                        if v[2]:
                            exist_col.append(v[1])
                    exist_col = list(set(exist_col))
                    rest_col = [v for v in self.column_names if v not in exist_col]
                    possible_combination = list(itertools.combinations(rest_col, len(new_pars)))
                    for comb in possible_combination:
                        par_permutes = itertools.permutations(new_pars)
                        for permute in par_permutes:
                            tmp = []
                            for pid,par in enumerate(permute):
                                tmp.append((par,comb[pid],True))
                                for col in self.column_names:
                                    if col!=comb[pid]:
                                        tmp.append((par, col, False))
                            cache.append(tmp)

                    for item in cache:
                        all_possible.append(vv + item)
                        # for assign in old_assginment:
                        #     all_possible.append(assign+item)
            return all_possible

    def find_satisfied_assignment(self,new_assignments):
        satisfied_assignment = []
        for v in new_assignments:
            if self.satisfy(v):
                satisfied_assignment.append(v)
        return satisfied_assignment

    def satisfy(self,assignment):
        #may contain multiple conditions and corresponding results
        flags = []
        cond_flag = []
        for id,condition in enumerate(self.conditions):
            flag = True
            for item in condition:
                item = (item['row'], item['column'],item['value'])
                if item not in assignment:
                    flag=False
            cond_flag.append(flag)
            for item in self.results[id]:
                item = (item['row'], item['column'], item['value'])
                if item not in assignment:
                    flag=False
            flags.append(flag)

        cond_satisfy = any(cond_flag)
        cond_res_flag = any(flags)
        # if self.__class__.__name__ == 'IFTHEN':
        #     return (not cond_satisfy) or cond_res_flag
        # else:
        return cond_res_flag

    def find_option_assignment(self):
        all_assignments = []
        for i, cond in enumerate(self.conditions):
            all_assignments.append(self.conditions[i]+self.results[i])
        return all_assignments



'''
tmp = {'participant': res.group(),
                           'position': (c2t[min(res.span())], c2t[max(res.span()) - 1]),
                           'type': 'row', 'idx': pidx}
'''
def append_ent(par_lst,col_lst,arg1,arg2):
    if arg1['type']=='row':
        par_lst.append(arg1['participant'])
    elif arg1['type']=='column':
        col_lst.append(arg1['participant'])
    if arg2['type']=='row':
        par_lst.append(arg2['participant'])
    elif arg2['type']=='column':
        col_lst.append(arg2['participant'])
    par_lst = list(set(par_lst))
    col_lst = list(set(col_lst))
    return par_lst,col_lst



class TO(Rule):
    def __init__(self,arg1,arg2,negation,all_pars,all_cols):
        super(TO, self).__init__(arg1,arg2,all_pars,all_cols)
        self.arg1 = arg1
        self.arg2 = arg2
        self.conditions = [[]]
        row_idx = arg1['participant'] if arg1['type'] == 'row' else arg2['participant']
        col_idx = arg1['participant'] if arg1['type'] == 'column' else arg2['participant']
        self.participants,self.columns = append_ent(self.participants,self.columns,arg1,arg2)
        v = False if negation else True
        self.results.append([{'row':row_idx,'column':col_idx,'value':v}])
        # print(self.results)

class DIFFERENT(Rule):
    def __init__(self,arg1,arg2,negation,all_pars,all_cols):

        super(DIFFERENT, self).__init__(arg1,arg2,all_pars,all_cols)

        assert(arg1['type']==arg2['type']),'arg1: {}, arg2:{}'.format(arg1,arg2)
        self.participants,self.columns = append_ent(self.participants,self.columns,arg1,arg2)
        if arg1['type'] == 'row': 
            for cid,col in enumerate(all_cols):
                for v in [1,-1]:
                        self.conditions.append([{'row':arg1['participant'],'column':col,'value':(v)>0}])
                        self.results.append([{'row': arg2['participant'], 'column': col, 'value':(v)<0 }])
                        self.conditions.append([{'row': arg2['participant'], 'column': col, 'value': (-v)>0}])
                        self.results.append([{'row': arg1['participant'], 'column': col, 'value': (-v)<0}])
        elif arg1['type'] == 'column':
            for rid,row in enumerate(all_pars):
                for v in [1,-1]:
                        self.conditions.append([{'row':row,'column':arg1['participant'],'value':(v)>0}])
                        self.results.append([{'row': row, 'column': arg2['participant'], 'value':(v)<0 }])
                        self.conditions.append([{'row': row, 'column': arg2['participant'], 'value': (-v)>0}])
                        self.results.append([{'row': row, 'column': arg1['participant'], 'value': (-v)<0}])



class SAME(Rule):
    def __init__(self,arg1,arg2,negation,all_pars,all_cols):
        super(SAME, self).__init__(arg1,arg2,all_pars,all_cols)
        assert (arg1['type'] == arg2['type']), 'arg1: {}, arg2:{}'.format(arg1, arg2)
        self.participants, self.columns = append_ent(self.participants, self.columns, arg1, arg2)
        if arg1['type'] == 'row':
            for cid, col in enumerate(all_cols):
                for v in [1,-1]:
                    self.conditions.append([{'row': arg1['participant'], 'column': col, 'value': (v)>0}])
                    self.results.append([{'row': arg2['participant'], 'column': col, 'value': (v)>0}])
                    self.conditions.append([{'row': arg2['participant'], 'column': col, 'value': (v)<0}])
                    self.results.append([{'row': arg1['participant'], 'column': col, 'value': (v)<0}])
        elif arg1['type'] == 'column':
            for rid, row in enumerate(all_pars):
                for v in [1,-1]:
                    self.conditions.append([{'row': row, 'column': arg1['participant'], 'value': (v)>0}])
                    self.results.append([{'row': row, 'column': arg2['participant'], 'value': (v)>0}])
                    self.conditions.append([{'row': row, 'column': arg2['participant'], 'value': (v)<0}])
                    self.results.append([{'row': row, 'column': arg1['participant'], 'value': (v)<0}])

class NEXT(Rule):
    def __init__(self,arg1,arg2,negation,all_pars,all_cols):
        super(NEXT, self).__init__(arg1,arg2,all_pars,all_cols)
        # print(arg1,arg2)
        assert (arg1['type'] == arg2['type']), 'arg1: {}, arg2:{}'.format(arg1, arg2)
        self.participants, self.columns = append_ent(self.participants, self.columns, arg1, arg2)
        #the column must be sorted

    def satisfy(self,assignment):
        arg1_pos, arg2_pos = None, None
        for item in assignment:
            if self.arg1['participant'] in item and item[2]:
                arg1_pos = self.column_names.index(item[1])
            if self.arg2['participant'] in item and item[2]:
                arg2_pos = self.column_names.index(item[1])
        if arg1_pos is not None and arg2_pos is not None:
            return arg2_pos == arg1_pos-1
        elif arg1_pos == 0:
            return False
        elif arg2_pos == len(self.column_names) - 1:
            return False
        else:
            return True

class LAST(NEXT):
    def __init__(self,arg1,arg2,negation,all_pars,all_cols):
        super(LAST, self).__init__(arg2,arg1,negation,all_pars,all_cols)

class ADJACENT(Rule):
    def __init__(self,arg1,arg2,negation,all_pars,all_cols):
        super(ADJACENT, self).__init__(arg1,arg2,all_pars,all_cols)
        # print(arg1,arg2)
        assert (arg1['type'] == arg2['type']), 'arg1: {}, arg2:{}'.format(arg1, arg2)
        self.participants, self.columns = append_ent(self.participants, self.columns, arg1, arg2)
        #the column must be sorted

    def satisfy(self,assignment):
        arg1_pos, arg2_pos = None, None
        for item in assignment:
            if self.arg1['participant'] in item and item[2]:
                arg1_pos = self.column_names.index(item[1])
            if self.arg2['participant'] in item and item[2]:
                arg2_pos = self.column_names.index(item[1])
        if arg1_pos is not None and arg2_pos is not None:
            return arg2_pos == arg1_pos-1 or arg2_pos == arg1_pos+1
        else:
            return True


class AFTER(Rule):
    def __init__(self,arg1,arg2,negation,all_pars,all_cols):
        super(AFTER, self).__init__(arg1,arg2,all_pars,all_cols)
        # assert (arg1['type'] == arg2['type']), 'arg1: {}, arg2:{}'.format(arg1, arg2)
        if arg1['type'] == arg2['type']:
            self.process_type = 'same_type'
        else:
            self.process_type = 'different_type'
        self.participants, self.columns = append_ent(self.participants, self.columns, arg1, arg2)

    def satisfy(self,assignment):
        if self.process_type == 'same_type':
            arg1_pos, arg2_pos = None,None
            for item in assignment:
                if self.arg1['participant'] in item and item[2]:
                    arg1_pos = self.column_names.index(item[1])
                if self.arg2['participant'] in item and item[2]:
                    arg2_pos = self.column_names.index(item[1])
            if arg1_pos is not None and arg2_pos is not None:
                return arg2_pos < arg1_pos
            elif arg1_pos==0:
                return False
            elif arg2_pos==len(self.column_names)-1:
                return False
            else:
                return True
        elif self.process_type=='different_type':
            arg1_pos, arg2_pos = None, None
            for item in assignment:
                if self.arg1['participant'] in item and item[2]:
                    rest_ent = item[1] if item[1] != self.arg1['participant'] else item[0]
                    arg1_pos = self.column_names.index(rest_ent) if self.arg1['type'] == 'row' else self.participants_name.index(rest_ent)

                if self.arg2['participant'] in item and item[2]:
                    rest_ent = item[1] if item[1] != self.arg2['participant'] else item[0]
                    arg2_pos = self.column_names.index(rest_ent) if self.arg2['type'] == 'row' else self.participants_name.index(rest_ent)
            if arg1_pos is not None and arg2_pos is not None:
                return arg2_pos < arg1_pos
            elif arg1_pos == 0:
                return False
            elif arg2_pos == len(self.column_names) - 1:
                return False
            else:
                return True

class BEFORE(AFTER):
    def __init__(self,arg1,arg2,negation,all_pars,all_cols):
        super(BEFORE, self).__init__(arg2,arg1,negation,all_pars,all_cols)



class BeforeEqual(Rule):
    def __init__(self,arg1,arg2,negation,all_pars,all_cols):
        super(BeforeEqual, self).__init__(arg1,arg2,all_pars,all_cols)
        # assert (arg1['type'] == arg2['type']), 'arg1: {}, arg2:{}'.format(arg1, arg2)
        if arg1['type'] != arg2['type']:
            self.process_type = 'same_type'
        else:
            self.process_type = 'different_type'
        self.participants, self.columns = append_ent(self.participants, self.columns, arg1, arg2)

    def satisfy(self,assignment):
        if self.process_type == 'same_type':
            arg1_pos, arg2_pos = None,None
            for item in assignment:
                if self.arg1['participant'] in item and item[2]:
                    arg1_pos = self.column_names.index(item[1])
                if self.arg2['participant'] in item and item[2]:
                    arg2_pos = self.column_names.index(item[1])
            if arg1_pos is not None and arg2_pos is not None:
                return arg2_pos >= arg1_pos
            elif arg2_pos==0:
                return False
            elif arg1_pos==len(self.column_names)-1:
                return False
            else:
                return True
        elif self.process_type=='different_type':
            arg1_pos, arg2_pos = None, None
            for item in assignment:
                if self.arg1['participant'] in item and item[2]:
                    rest_ent = item[1] if item[1] != self.arg1['participant'] else item[0]
                    arg1_pos = self.column_names.index(rest_ent) if self.arg1['type'] == 'row' else self.participants_name.index(rest_ent)

                if self.arg2['participant'] in item and item[2]:
                    rest_ent = item[1] if item[1] != self.arg2['participant'] else item[0]
                    arg2_pos = self.column_names.index(rest_ent) if self.arg2['type'] == 'row' else self.participants_name.index(rest_ent)
            if arg1_pos is not None and arg2_pos is not None:
                return arg2_pos >= arg1_pos
            elif arg2_pos == 0:
                return False
            elif arg1_pos == len(self.column_names) - 1:
                return False
            else:
                return True

class LastNum(Rule):
    def __init__(self, close_ent, num, all_pars, all_cols):
        super(LastNum, self).__init__(None, None, all_pars, all_cols)
        self.ent = close_ent
        self.num = num
        if self.ent['participant'] not in self.participants:
            self.participants.append(self.ent['participant'])
    def satisfy(self,assignment):
        ent_pos = None
        # print(assignment,self.ent)
        for item in assignment:
            if self.ent['participant'] in item and item[2]:
                ent_pos = self.column_names.index(item[1])
        # print(range(len(self.column_names)-self.num,len(self.column_names)))
        # print(self.num)
        if (ent_pos in list(range(len(self.column_names)-self.num,len(self.column_names)))): #or ent_pos is None:
            # print(ent_pos, assignment)
            return True
        else:
            return False

class FirstNum(Rule):
    def __init__(self, close_ent, num, all_pars, all_cols):
        super(FirstNum, self).__init__(None, None, all_pars, all_cols)
        self.ent = close_ent
        self.num = num
        if self.ent['participant'] not in self.participants:
            self.participants.append(self.ent['participant'])
    def satisfy(self,assignment):
        ent_pos = None

        for item in assignment:
            if self.ent['participant'] in item and item[2]:
                ent_pos = self.column_names.index(item[1])
        if (ent_pos in range(0,self.num)):# or ent_pos is None:
            # print(ent_pos,assignment)
            return True
        else:
            return False

# class BEFORE(NEXT):
#     def __init__(self,arg1,arg2,all_pars,all_cols):
#         super(BEFORE, self).__init__(arg2,arg1,all_pars,all_cols)
class AND(Rule):
    def __init__(self,prev_rule_set,after_rule_set,all_pars,all_cols):
        super(AND, self).__init__(None,None,all_pars,all_cols)
        self.prev_rule_set = prev_rule_set
        self.after_rule_set = after_rule_set
        for rule in self.prev_rule_set+self.after_rule_set:
            self.participants.extend(rule.participants)
            self.columns.extend(rule.columns)
        self.participants = list(set((self.participants)))
        self.columns = list(set(self.columns))

    def satisfy(self,assignment):
        count = 0
        prev_satisfy = []
        after_satisfy = []
        for rule in self.prev_rule_set:
            if rule.satisfy(assignment):
                prev_satisfy.append(True)
            else:
                prev_satisfy.append(False)
        for rule in self.after_rule_set:
            if rule.satisfy(assignment):
                after_satisfy.append(True)
            else:
                after_satisfy.append(False)
        if all(prev_satisfy) and all(after_satisfy):
            return True
        else:
            return False
class OR(Rule):
    def __init__(self,prev_rule_set,after_rule_set,all_pars,all_cols):
        super(OR, self).__init__(None,None,all_pars,all_cols)
        self.prev_rule_set = prev_rule_set
        self.after_rule_set = after_rule_set
        for rule in self.prev_rule_set+self.after_rule_set:
            self.participants.extend(rule.participants)
            self.columns.extend(rule.columns)
        self.participants = list(set((self.participants)))
        self.columns = list(set(self.columns))

    def satisfy(self,assignment):
        count = 0
        prev_satisfy = []
        after_satisfy = []
        for rule in self.prev_rule_set:
            if rule.satisfy(assignment):
                prev_satisfy.append(True)
            else:
                prev_satisfy.append(False)
        for rule in self.after_rule_set:
            if rule.satisfy(assignment):
                after_satisfy.append(True)
            else:
                after_satisfy.append(False)
        if all(prev_satisfy):
            count+=1
        if all(after_satisfy):
            count+=1
        return count == 1

class UNLESS(Rule):
    def __init__(self,unless_rules,rest_rules,all_parts,all_cols):
        super(UNLESS, self).__init__(None,None,all_parts,all_cols)
        self.unless_rule_set = unless_rules
        self.rest_rule_set = rest_rules
        for rule in self.unless_rule_set+self.rest_rule_set:
            self.participants.extend(rule.participants)
            self.columns.extend(rule.columns)
        self.participants = list(set(self.participants))
        self.columns = list(set(self.columns))

    def satisfy(self,assignment):
        unless_satisfy = []
        res_satisfy = []
        for rule in self.unless_rule_set:
            unless_satisfy.append(rule.satisfy(assignment))
        for rule in self.rest_rule_set:
            res_satisfy.append(rule.satisfy(assignment))
        if all(unless_satisfy):
            return True
        elif (not all(unless_satisfy)) and all(res_satisfy):
            return True
        else:
            return False


class NEITHER(Rule):
    def __init__(self,neither_rules,nor_rules,all_parts,all_cols):
        super(NEITHER, self).__init__(None,None,all_parts,all_cols)
        self.neither_rule_set = neither_rules
        self.nor_rule_set = nor_rules
        for rule in self.neither_rule_set+self.nor_rule_set:
            self.participants.extend(rule.participants)
            self.columns.extend(rule.columns)
        self.participants = list(set(self.participants))
        self.columns = list(set(self.columns))

    def satisfy(self,assignment):
        neither_satisfy = []
        nor_satisfy = []
        for rule in self.neither_rule_set:
            neither_satisfy.append(rule.satisfy(assignment))
        for rule in self.nor_rule_set:
            nor_satisfy.append(rule.satisfy(assignment))
        if not all(neither_satisfy) and not all(nor_satisfy):
            return True
        else:
            return False
class IFF(Rule):
    def __init__(self,p_rules,q_rules,all_parts,all_cols):
        super(IFF,self).__init__(None,None,all_parts,all_cols)
        self.p_rule_set = p_rules
        self.q_rule_set = q_rules
        # print(self.if_rule_set+self.then_rule_set)
        for rule in self.p_rule_set+self.q_rule_set:
            self.participants.extend(rule.participants)
            self.columns.extend(rule.columns)
        self.participants = list(set(self.participants))
        self.columns = list(set(self.columns))

    def satisfy(self,assignment):
        cond_satisfy = []
        res_satisfy = []
        for rule in self.p_rule_set:
            cond_satisfy.append(rule.satisfy(assignment))
        for rule in self.q_rule_set:
            res_satisfy.append(rule.satisfy(assignment))
        flag = all(cond_satisfy) and all(res_satisfy)
        if ((not any(cond_satisfy)) and (not any(res_satisfy))):
            return True
        else:
            return flag

class IFTHEN(Rule):
    def __init__(self,if_rules,then_rules,all_parts,all_cols):
        super(IFTHEN,self).__init__(None,None,all_parts,all_cols)
        '''
        complicated:
        if if_rules_set contains 2 rule, and each rule have n,m (cond,res) pair, the condition in if rule set should have n*m conditions
        and then_rules_set contains 2 rule, each rule have p,q (cond,res) pair， the final condition number should be n*m*p*q (cummulatives of all condition+result in ifrule and all condition in then rule)
            have n*m*p*q (cond,res) results pair, each result is the concatenation of result in pair(p_i,q_j) and its repeated n*m times
        '''
        self.if_rule_set = if_rules
        self.then_rule_set = then_rules
        # print(self.if_rule_set+self.then_rule_set)
        for rule in self.if_rule_set+self.then_rule_set:
            self.participants.extend(rule.participants)
            self.columns.extend(rule.columns)
            self.participants = list(set(self.participants))
            self.columns = list(set(self.columns))


    def satisfy(self,assignment):

        cond_satisfy = []
        res_satisfy = []
        for rule in self.if_rule_set:
            cond_satisfy.append(rule.satisfy(assignment))
        for rule in self.then_rule_set:
            res_satisfy.append(rule.satisfy(assignment))
        flag = cond_satisfy and all(cond_satisfy)
        if flag:
            return all(res_satisfy)
        else:
            return True

