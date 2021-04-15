import re,copy,nltk
from modify_option import analyze_question
import numpy as np
from scipy.special import softmax
numbers = {
            'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 'eleven': 11,'twelve': 12
        }
negation_words = [wd for wd in ['not', 'nor', 'no', 'never', "didn't", "won't", "wasn't", "isn't",
                                      "haven't", "weren't", "won't", 'neither', 'none', 'unable', 'outside',
                                      'unable', 'fail', 'cannot', 'except', 'CANNOT','EXCEPT','Neither','neither','false']]

def find_nearest_ent(kws,question,rows,columns):
    for item in kws:
        pos = question.find(item)
        min_dis, min_ent = 99999, None
        for ent in rows + columns:
            ent_pos = question.find(ent, pos)
            if ent_pos != -1:
                dis = abs(ent_pos - pos)
                if dis < min_dis:
                    min_dis = dis
                    min_ent = ent
            if min_ent:
                return min_ent
        return min_ent
def choose_question_type(question,leaf_nodes,rows,columns,problem_type,question_tags,all_rules):
    if 'must be true' in question or any(['must be true' in tag for tag in question_tags]):
        Q = MUSTTRUE(problem_type,question,leaf_nodes,all_rules)
    elif 'must be false' in question or any(['must be false' in tag for tag in question_tags]):
        Q = MUSTFALSE(problem_type, question, leaf_nodes, all_rules)
    elif 'maximum' in question:
        pos = question.find('maximum')
        min_dis,min_ent = 99999, None
        for ent in rows+columns:
            ent_pos = question.find(ent,pos)
            if ent_pos!=-1:
                dis = abs(ent_pos-pos)
                if dis < min_dis:
                    min_dis = dis
                    min_ent = ent
        Q = MAXIMUM(problem_type,question,leaf_nodes,all_rules,min_ent)
    elif any([item for item in ['How many','count','number of'] if item in question]):
        Q = None
        min_ent = find_nearest_ent(['How many','count','number of'],question,rows,columns)
        if min_ent:
            Q = COUNT(problem_type,question, leaf_nodes, all_rules,min_ent)
        if not Q:
            Q = QuestionType(problem_type, question, leaf_nodes,all_rules)
    elif any([item for item in ['list','order'] if item in question]):
        Q = QuestionType(problem_type,question,leaf_nodes,all_rules)
    elif any([item for item in ['could be false'] if item in question]) or any(['could be false' in tag for tag in question_tags]):
        Q = CouldFalse(problem_type, question, leaf_nodes,all_rules)
    elif any([item for item in ['earliest'] if item in question]):
        Q = None
        min_ent = find_nearest_ent(['earliest'], question, rows, columns)
        if min_ent:
            Q = EARLIEST(problem_type,question,leaf_nodes,all_rules,min_ent,columns)
        if not Q:
            Q = QuestionType(problem_type, question, leaf_nodes, all_rules)
    else:
        Q = MUSTTRUE(problem_type, question, leaf_nodes,all_rules)
    #judge based on tags:
    # if any(['must be true' in tag for tag in question_tags]):
    #     Q = MUSTTRUE(problem_type, question, leaf_nodes)
    # elif any(['could be true' in tag for tag in question_tags]):
    #     Q = QuestionType(problem_type, question, leaf_nodes)
    return Q


class QuestionType():
    def __init__(self,problem_type,question,leaf_nodes,rules):
        super(QuestionType, self).__init__()
        self.problem_type = problem_type
        self.question = question
        self.leaf_nodes = leaf_nodes
        self.useful_part = self.extract_useful_part()
        self.polarity = self.contain_negation()
        self.all_rules = rules


    def contain_negation(self):
        # print(self.useful_part)
        if ([wd for wd in negation_words if wd in self.useful_part+' ']):
            v = False
        else:
            v = True
        return v

    def select_answers(self,answers, opt_assigns, opt_funcs):
        # print(self.polarity)
        all_assign_scores = []
        for i,opt_assign in enumerate(opt_assigns):
            ans = answers[i]
            score = self.score_calculating(opt_assign,ans)
            all_assign_scores.append((score,i))

        # print(sorted_score)
        all_func_score = []
        for i, opt_func in enumerate(opt_funcs):
            score = self.func_score_calculating(opt_func)
            all_func_score.append((score,i))

        # all_pure_opt_score = []
        # for i, opt_assign in enumerate(opt_assigns):
        #     score = []
        #     for assign in opt_assign:
        #         tmp_score = 0
        #         assign = [(item['row'], item['column'], item['value']) for item in assign]
        #         for func in self.all_rules:
        #             if func.satisfy(assign):
        #                 tmp_score+=1
        #                 # print(type(func),score)
        #             else:
        #                 tmp_score -= 1
        #         if self.all_rules:
        #             score.append(tmp_score/len(self.all_rules))
        #         else:
        #             score.append(0)
        #     score = max(score) if score else 0
        #     # print(opt_assign,score)
        #     all_pure_opt_score.append((score,i))
        # overall_score = []

        overall_score = [(all_assign_scores[i][0]+all_func_score[i][0],i) for i in range(len(all_assign_scores))]
        # if all([all_assign_scores[i][0]==0 for i in range(len(all_assign_scores))]):
        #     overall_score = all_func_score
        # else:
        #     overall_score = all_assign_scores
        # overall_score = [(all_func_score[i][0], i) for i in range(len(all_assign_scores))]
        if self.polarity:
            sorted_score = sorted(overall_score, key=lambda k: k[0], reverse=True)
        else:
            sorted_score = sorted(overall_score, key=lambda k: k[0], reverse=False)
        return sorted_score

    def extract_useful_part(self):
        attention_word = ['then', 'which', 'what',',']
        self.question = self.question.replace(':',' ')
        for attn_word in attention_word:
            if attn_word in self.question:
                return self.question[self.question.find(attn_word) + len(attn_word):]
            else:
                return self.question

    def func_node_score(self,opt_func):
        all_node_score = []
        for nid, node in enumerate(self.leaf_nodes):
            node_score = []
            for func in opt_func:
                res = 1 if func.satisfy(node.assignment) else -1
                node_score.append(res)
            if node_score:
                node_score = sum(node_score)
            else:
                node_score = 0
            all_node_score.append(node_score)
        return all_node_score

    def func_score_calculating(self,opt_func):
        all_node_score = self.func_node_score(opt_func)
        if all_node_score:
            return max(all_node_score)
        else:
            return 0

    def assign_node_score(self,opt_assign,answer=None):
        all_score = []
        for aid, single_assign in enumerate(opt_assign):
            nodes_score = []
            for nid, node in enumerate(self.leaf_nodes):
                tmp_score = 0
                for item in single_assign:
                    t = (item['row'], item['column'], item['value'])
                    neg_t = (item['row'], item['column'], False) if item['value'] else (
                    item['row'], item['column'], True)
                    if t in node.assignment:
                        tmp_score += 1
                    elif neg_t in node.assignment:
                        tmp_score -= 1
                if single_assign:
                    tmp_score = tmp_score / len(single_assign)
                else:
                    tmp_score = 0
                # if tmp_score > max_score:
                #     max_score = tmp_score
                #     max_idx = nid
                nodes_score.append(tmp_score)

            all_score.append((nodes_score))
        return all_score

    def score_calculating(self,opt_assign,answer=None):
        all_nodes_score = self.assign_node_score(opt_assign,answer)
        all_score = [max(score) for score in all_nodes_score if score]
        if all_score:
            return max(all_score)
        else:
            return 0

class CouldFalse(QuestionType):
    def __init__(self,problem_type,question,leaf_nodes,all_rules):
        super(CouldFalse, self).__init__(problem_type,question,leaf_nodes,all_rules)

    def score_calculating(self, opt_assign,answer=None):
        max_score, max_idx = -999, None
        # print(opt_assign)
        all_nodes_score = self.assign_node_score(opt_assign,answer)
        all_score = [min(score) for score in all_nodes_score if score]
        return min(all_score) if all_score else 0
    def func_score_calculating(self,opt_func):
        all_score = self.func_node_score(opt_func)
        return sum(all_score)

class MUSTTRUE(QuestionType):
    def __init__(self,problem_type,question,leaf_nodes,all_rules):
        super(MUSTTRUE, self).__init__(problem_type,question,leaf_nodes,all_rules)

    def score_calculating(self, opt_assign,answer=None):
        max_score, max_idx = -999, None
        # print(opt_assign)
        all_nodes_score = self.assign_node_score(opt_assign,answer)
        all_score = [sum(score)/len(score) for score in all_nodes_score if score]

        return max(all_score) if all_score else 0
    def func_score_calculating(self,opt_func):
        all_score = self.func_node_score(opt_func)
        if all_score:
            return sum(all_score)/len(all_score)
        else:
            return 0

class MUSTFALSE(QuestionType):
    def __init__(self, problem_type, question, leaf_nodes, all_rules):
        super(MUSTFALSE, self).__init__(problem_type, question, leaf_nodes, all_rules)

    def score_calculating(self, opt_assign, answer=None):
        max_score, max_idx = -999, None
        # print(opt_assign)
        all_nodes_score = self.assign_node_score(opt_assign, answer)
        all_score = [sum(score) / len(score) for score in all_nodes_score if score]

        return max(all_score) if all_score else 0

    def func_score_calculating(self, opt_func):
        all_score = self.func_node_score(opt_func)
        if all_score:
            return sum(all_score) / len(all_score)
        else:
            return 0

class MAXIMUM(QuestionType):
    def __init__(self,problem_type,question,leaf_nodes,all_rules,ent):
        super(MAXIMUM, self).__init__(problem_type,question,leaf_nodes,all_rules)
        self.ent = ent
    def score_calculating(self,opt_assign,answer):
        words = nltk.word_tokenize(answer)
        number = None
        for word in words:
            if word in numbers.keys():
                number = numbers[word]
                break
        ent_counts = []
        if number is not None:
            for node in self.leaf_nodes:
                count = 0
                # print(len(node.assignment))
                # print(len(list(set(node.assignment))))
                for item in node.assignment:
                    if self.ent in item and item[2]:
                        count += 1
                ent_counts.append(count)
        # print(ent_counts)
        if ent_counts:
            max_num = max(ent_counts)
        else:
            max_num = 0
        if number:
            dis = abs(max_num-number)
            score = -dis
        else:
            score = -999
        return score

class EARLIEST(QuestionType):
    def __init__(self,problem_type,question,leaf_nodes,all_rules,ent,columns):
        super(EARLIEST, self).__init__(problem_type,question,leaf_nodes,all_rules)
        self.ent = ent
        self.columns = columns
    def score_calculating(self,opt_assign,answer=None):
        words = nltk.word_tokenize(answer)
        number = None
        for word in words:
            if word in numbers.keys():
                number = numbers[word]
                break
        min_col = 999
        if number is not None:
            for node in self.leaf_nodes:
                for item in node.assignment:
                    if self.ent==item[0] and item[2]:
                        col = self.columns.index(item[1])
                        if col < min_col:
                            min_col = col
        if number and min_col!=999:
            dis = abs(min_col+1-number)
            score = -dis
        else:
            score = 0
        return score

class COUNT(QuestionType):
    def __init__(self,problem_type,question,leaf_nodes,all_rules,ent):
        super(COUNT, self).__init__(problem_type,question,leaf_nodes,all_rules)
        self.ent = ent

    def score_calculating(self,opt_assign,answer):
        words = nltk.word_tokenize(answer)
        number = None
        for word in words:
            if word in numbers.keys():
                number = numbers[word]
                break
        ent_counts = []
        all_occur = []
        if number is not None:
            for node in self.leaf_nodes:
                count = []
                # print(len(node.assignment))
                # print(len(list(set(node.assignment))))
                for item in node.assignment:
                    if self.ent in item and item[2]:
                        count.append(item)
                all_occur.extend(count)
                ent_counts.append(len(count))
        if self.problem_type == 'ordering':
            occur_set = list(set(all_occur))
            pred = len(occur_set)
        elif self.problem_type == 'grouping':
            if ent_counts:
                pred = max(ent_counts)
            else:
                pred = 0
        if number:
            dis = abs(pred-number)
            score = -dis
        else:
            score = -999
        return score



