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

    def __init__(self, example_id,contexts, label=None):
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
        self.contexts = contexts
        self.label = label


class InputFeatures(object):
    def __init__(self, example_id, choices_features, label):
        self.example_id = example_id
        self.choices_features = choices_features
        # self.choices_features = [
        #     {"input_ids": input_ids,'lengths':lengths}
        #     for input_ids,lengths in choices_features
        # ]
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

    def get_train_examples(self, data_dir,type,text_field):
        """See base class."""
        file_name = os.path.join(data_dir, type)
        logger.info("LOOKING AT {} train".format(file_name))
        # return self._create_examples(self._read_json(os.path.join(data_dir, "train.json")))

        return self._create_examples(self._read_json(file_name),text_field)

    def get_dev_examples(self, data_dir,type,text_field):
        """See base class."""
        file_name = os.path.join(data_dir, type)
        logger.info("LOOKING AT {} dev".format(file_name))
        # return self._create_examples(self._read_json(os.path.join(data_dir, "val.json")))
        return self._create_examples(self._read_json(file_name),text_field)

    def get_test_examples(self, data_dir, type, text_field):
        logger.info("LOOKING AT {} test".format(data_dir))
        # return self._create_examples(self._read_json(os.path.join(data_dir, "test.json")))
        # return self._create_examples(self._read_json(os.path.join(data_dir, "test_lr.json")))
        return self._create_examples(self._read_json(os.path.join(data_dir, type)),text_field)

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

    def _create_examples(self, lines, text_field):
        """Creates examples for the training and dev sets."""
        examples = []
        texts = []
        for d in lines:
            context = d['context']
            question = d['question']
            answers = d['answers'] #[text_field.preprocess(x) for x in d['answers']]
            label = d['label']
            id_string = d['id_string']
            all_context = [text_field.preprocess(context+' '+question+' '+ans) for ans in answers]
            if len(answers)==5:
                examples.append(
                    InputExample(
                        example_id = id_string,
                        contexts = all_context,
                        # question = text_field.preprocess(question),
                        # contexts=text_field.preprocess(context),#[context, context, context, context, context],
                        # endings=[answers[0], answers[1], answers[2], answers[3], answers[4]],
                        label = label
                        )
                    )
            texts.extend(all_context)
        return examples,texts

import re
def clean_string(string):
    return re.sub(r'[^a-zA-Z0-9,.\'!?]+', ' ', string)
def convert_examples_to_features(
    examples: List[InputExample],
    label_list: List[str],
    max_length: int,
    text_field,
) -> List[InputFeatures]:
    """
    Loads a data file into a list of `InputFeatures`
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert examples to features"):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        #for ending_idx, (context, question, ending) in enumerate(zip(example.contexts, example.question, example.endings)):
        padded_contexts, lengths = text_field.pad(example.contexts)
        # print(len(padded_contexts))

        tmp = tuple([padded_contexts, lengths])
        input_ids, lengths = text_field.numericalize(tmp)
        # print(input_ids.shape)
        # print(len(tmp[0]),len(lengths),text_field.use_vocab,text_field.sequential)
        choices_features={'input_ids':input_ids,'lengths':lengths}#[(input_ids[i],lengths[i]) for i in range((input_ids.size(0)))]
        label = label_map[example.label]

        if ex_index < 2:
            logger.info("*** Example ***")
            logger.info("race_id: {}".format(example.example_id))
            for choice_idx, (context) in enumerate(padded_contexts):
                logger.info("choice: {}".format(choice_idx))
                logger.info("context: {}".format(context))
                logger.info("input_ids: {}".format(input_ids[choice_idx,:]))
                logger.info("label: {}".format(label))

        features.append(InputFeatures(example_id=example.example_id, choices_features=choices_features, label=label))

    return features


processors = {"lsat": LSATProcessor}

