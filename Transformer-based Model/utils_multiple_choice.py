# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Multiple choice fine-tuning: utilities to work with multiple choice tasks of reading comprehension  """


import csv
import glob
import json
import logging
import os
from typing import List

import tqdm

from transformers import PreTrainedTokenizer


logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for multiple choice"""

    def __init__(self, example_id, question, contexts, endings, label=None):
        """Constructs a InputExample.

        Args:
            example_id: Unique id for the example.
            contexts: list of str. The untokenized text of the first sequence (context of corresponding question).
            question: string. The untokenized text of the second sequence (question).
            endings: list of str. multiple choice's options. Its length must be equal to contexts' length.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.example_id = example_id
        self.question = question
        self.contexts = contexts
        self.endings = endings
        self.label = label


class InputFeatures(object):
    def __init__(self, example_id, choices_features, label):
        self.example_id = example_id
        self.choices_features = [
            {"input_ids": input_ids, "input_mask": input_mask, "segment_ids": segment_ids}
            for input_ids, input_mask, segment_ids in choices_features
        ]
        self.label = label


class DataProcessor(object):
    """Base class for data converters for multiple choice data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


class LSATProcessor(DataProcessor):
    """Processor for the LSAT data set."""

    def get_train_examples(self, data_dir,type=''):
        """See base class."""
        file_name = os.path.join(data_dir, "train_%s.json"%type)
        logger.info("LOOKING AT {} train".format(file_name))
        # return self._create_examples(self._read_json(os.path.join(data_dir, "train.json")))

        return self._create_examples(self._read_json(file_name))

    def get_dev_examples(self, data_dir,type=''):
        """See base class."""
        file_name = os.path.join(data_dir, "val_%s.json" % type)
        logger.info("LOOKING AT {} dev".format(file_name))
        # return self._create_examples(self._read_json(os.path.join(data_dir, "val.json")))
        return self._create_examples(self._read_json(file_name))

    def get_test_examples(self, data_dir, type=""):
        logger.info("LOOKING AT {} test".format(data_dir))
        # return self._create_examples(self._read_json(os.path.join(data_dir, "test.json")))
        # return self._create_examples(self._read_json(os.path.join(data_dir, "test_lr.json")))
        return self._create_examples(self._read_json(os.path.join(data_dir, "test_"+ type +".json")))

    def get_test_examples_specific(self, data_dir, type):
        logger.info("LOOKING AT %s test_%s", data_dir, type)
        return self._create_examples(self._read_json(os.path.join(data_dir, "test_"+type+".json")))

    def get_labels(self):
        """See base class."""
        return [0, 1, 2, 3, 4]

    def _read_json(self, input_file):
        with open(input_file, "r") as f:
            lines = json.load(f)
        return lines

    def _create_examples(self, lines):
        """Creates examples for the training and dev sets."""
        examples = []
        for d in lines:
            context = d['context']
            question = d['question']
            answers = d['answers']
            label = d['label']
            id_string = d['id_string']
            examples.append(
                InputExample(
                    example_id = id_string,
                    question = question,
                    contexts=context,#[context, context, context, context, context],
                    endings=[answers[0], answers[1], answers[2], answers[3], answers[4]],
                    label = label
                    )
                )  
        return examples

import re
def clean_string(string):
    return re.sub(r'[^a-zA-Z0-9,.\'!?]+', ' ', string)
def convert_examples_to_features(
    examples: List[InputExample],
    label_list: List[str],
    max_length: int,
    tokenizer: PreTrainedTokenizer,
    pad_token_segment_id=0,
    pad_on_left=False,
    pad_token=0,
    mask_padding_with_zero=True,
) -> List[InputFeatures]:
    """
    Loads a data file into a list of `InputFeatures`
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert examples to features"):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        choices_features = []
        #for ending_idx, (context, question, ending) in enumerate(zip(example.contexts, example.question, example.endings)):
        for ending_idx, ending in enumerate(example.endings):
            # text_a = example.question+ ' '+ending
            # text_a = example.question
            text_a = example.contexts
            text_b = example.question + " " + ending

            # text_b = clean_string(' '.join(example.contexts.split()[:100]))
            # print('text_a is', text_a)
            # print('text_b is',text_b)
            # print('-----------------------')
            # text_a = context
            # text_b = ending

            # inputs = tokenizer.encode_plus(text_a, text_b, add_special_tokens=True, max_length=max_length,)
            inputs = tokenizer.encode_plus(text_a, text_b, add_special_tokens=True, max_length=max_length, pad_to_max_length=True, return_attention_mask=True,)
            # inputs = tokenizer.encode_plus(text_b, add_special_tokens=True, max_length=max_length, pad_to_max_length=True, return_attention_mask=True,)
            # inputs = tokenizer.encode_plus(text_a, add_special_tokens=True, max_length=max_length,)
            # inputs = tokenizer(ending, add_special_tokens=True, max_length=max_length)
            if "num_truncated_tokens" in inputs and inputs["num_truncated_tokens"] > 0:
                logger.info(
                    "Attention! you are cropping tokens (swag task is ok). "
                    "If you are training ARC and RACE and you are poping question + options,"
                    "you need to try to use a bigger max seq length!"
                )

            input_ids, attention_mask = inputs["input_ids"], inputs['attention_mask']
            token_type_ids = attention_mask
            # print(len(input_ids))
            # print(input_ids)
            # print(token_type_ids)
            # print("*"*80)

            # # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # # tokens are attended to.
            # attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
            #
            # # Zero-pad up to the sequence length.
            # padding_length = max_length - len(input_ids)
            # if pad_on_left:
            #     input_ids = ([pad_token] * padding_length) + input_ids
            #     attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            #     token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
            # else:
            #     input_ids = input_ids + ([pad_token] * padding_length)
            #     attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            #     token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

            assert len(input_ids) == max_length
            assert len(attention_mask) == max_length
            assert len(token_type_ids) == max_length
            choices_features.append((input_ids, attention_mask, token_type_ids))

        # print(tokenizer.sep_token, tokenizer.sep_token_id)
        # print(tokenizer.pad_token, tokenizer.pad_token_id)
        # print(tokenizer.bos_token, tokenizer.bos_token_id)
        # print(tokenizer.eos_token, tokenizer.eos_token_id)
        # print(tokenizer.unk_token, tokenizer.unk_token_id)
        # print(tokenizer.cls_token, tokenizer.cls_token_id)
        # print(tokenizer.mask_token, tokenizer.mask_token_id)
        # print(tokenizer.additional_special_tokens, tokenizer.additional_special_tokens_ids)
        # print(tokenizer.SPECIAL_TOKENS_ATTRIBUTES)
        # return

        label = label_map[example.label]

        # if ex_index < 2:
        #     logger.info("*** Example ***")
        #     logger.info("race_id: {}".format(example.example_id))
        #     for choice_idx, (input_ids, attention_mask, token_type_ids) in enumerate(choices_features):
        #         logger.info("choice: {}".format(choice_idx))
        #         logger.info("input_ids: {}".format(" ".join(map(str, input_ids))))
        #         logger.info("attention_mask: {}".format(" ".join(map(str, attention_mask))))
        #         logger.info("token_type_ids: {}".format(" ".join(map(str, token_type_ids))))
        #         logger.info("label: {}".format(label))

        features.append(InputFeatures(example_id=example.example_id, choices_features=choices_features, label=label,))

    return features


processors = {"lsat": LSATProcessor}

