import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
# from transformers.modeling_bert import BertLayerNorm
import torch.nn.functional as F
from transformers import RobertaConfig
from transformers import PreTrainedModel,RobertaModel#, RobertaPreTrainedModel
from allennlp.modules.attention import DotProductAttention
from allennlp.nn import util
from typing import Dict, Tuple, Sequence,Optional

class LSATLSTM(nn.Module):
    def __init__(self,vocab_size,input_size,hidden_size,batch_first=True,max_seq_length=30):
        super(LSATLSTM, self).__init__()
        self.batch_first = batch_first
        self.embedding = nn.Embedding(vocab_size,input_size)
        self.max_seq_length = max_seq_length
        self.rnn = torch.nn.GRU(input_size=input_size,hidden_size=hidden_size,batch_first=batch_first)
        # self.rnn = torch.nn.LSTM(input_size=input_size,hidden_size=hidden_size,batch_first=batch_first)
        self.fc = nn.Linear(hidden_size, 1)

        # self.rnn = NaiveLSTM(input_sz=input_size,hidden_sz=hidden_size)
        # self.rnn = rnn_util.LayerNormLSTM(input_size=input_size,hidden_size=hidden_size,num_layers=1,
        #                                    dropout=0,bidirectional=False,layer_norm_enabled=True)
    def forward(self,inputs,seq_lengths,labels):#,score):
        # print(inputs.shape,hidden.shape)
        flat_lengths = seq_lengths.view(-1)
        flat_inputs = inputs.view(flat_lengths.size(0),-1)
        num_choices = inputs.size(1)
        embedded_inputs = self.embedding(flat_inputs)
        # flat_inputs = embedded_inputs.view(-1, inputs.size(-1))

        # print(flat_inputs.shape,flat_lengths.shape)
        packed_inputs = torch.nn.utils.rnn.pack_padded_sequence(embedded_inputs,flat_lengths,batch_first=self.batch_first,enforce_sorted=False)
        # res , (hn,cn) = self.rnn(input=packed_inputs,delta=min_score)
        res, hn = self.rnn(packed_inputs)
        # res, (hn, cn) = self.rnn(packed_inputs)
        padded_res,_ = nn.utils.rnn.pad_packed_sequence(res,batch_first=self.batch_first,total_length=self.max_seq_length)#batch,max_seq_length,hidden
        # padded_gate,_ = nn.utils.rnn.pad_packed_sequence(gates, batch_first=self.batch_first,total_length=self.max_seq_length)
        # hn = torch.cat([hn[0,:,:],hn[1,:,:]],dim=-1)
        logits = self.fc(hn.squeeze(0))
        reshaped_logits = logits.view(-1, num_choices)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
        return (loss,) + (reshaped_logits,)
        # return hn.squeeze(0),padded_res
        # padded_res, _ = nn.utils.rnn.pad_packed_sequence(res,batch_first=self.batch_first)




