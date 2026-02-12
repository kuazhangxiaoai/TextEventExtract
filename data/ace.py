import os
import json
import xml.etree.ElementTree as ET

import torch
from markdown_it.rules_inline import entity
from tqdm import tqdm
import numpy as np
from bs4 import BeautifulSoup
import nltk
import re
from data.utils import get_files_from_dir, get_base_name, pad_2d, get_tokenizer, pad_3d, pad_4d
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class Parser:
    def __init__(self, sgm_path, apf_path):
        #self.path = path
        self.sgm_path = sgm_path
        self.apf_path = apf_path
        self.entity_mentions = []
        self.event_mentions = []
        self.sentences = []
        self.sgm_text = ''

        self.entity_mentions, self.event_mentions, self.relation_mentions = self.parse_xml(apf_path)
        self.sents_with_pos = self.parse_sgm(sgm_path)
        self.fix_wrong_position()

    @staticmethod
    def clean_text(text):
        return text.replace('\n', ' ')

    def get_data(self):
        data = []
        for sent in self.sents_with_pos:
            item = dict()

            item['sentence'] = self.clean_text(sent['text'])
            item['position'] = sent['position']
            text_position = sent['position']

            for i, s in enumerate(item['sentence']):
                if s != ' ':
                    item['position'][0] += i
                    break

            item['sentence'] = item['sentence'].strip()

            entity_map = dict()
            item['golden-entity-mentions'] = []
            item['golden-event-mentions'] = []
            item['golden-relation-mentions'] = []

            for entity_mention in self.entity_mentions:
                entity_position = entity_mention['position']
                if text_position[0] <= entity_position[0] and entity_position[1] <= text_position[1]:
                    item['golden-entity-mentions'].append({
                        'text': self.clean_text(entity_mention['text']),
                        'position': entity_position,
                        'entity-type': entity_mention['entity-type'],
                        'head': {
                            "text": self.clean_text(entity_mention['head']["text"]),
                            "position": entity_mention["head"]["position"]
                        },
                        "entity_id": entity_mention['entity-id']
                    })
                    entity_map[entity_mention['entity-id']] = entity_mention

            for event_mention in self.event_mentions:
                event_position = event_mention['trigger']['position']
                if text_position[0] <= event_position[0] and event_position[1] <= text_position[1]:
                    event_arguments = []
                    for argument in event_mention['arguments']:
                        try:
                            entity_type = entity_map[argument['entity-id']]['entity-type']
                        except KeyError:
                            print('[Warning] The entity in the other sentence is mentioned. This argument will be ignored.')
                            continue
                        event_arguments.append({
                            'role': argument['role'],
                            'position': argument['position'],
                            'entity-type': entity_type,
                            'text': self.clean_text(argument['text']),
                        })
                    item['golden-event-mentions'].append({
                        'trigger': event_mention['trigger'],
                        'arguments': event_arguments,
                        'position': event_position,
                        'event_type': event_mention['event_type'],
                    })
            for relation_mention in self.relation_mentions:
                relation_postion = relation_mention['position']
                if text_position[0] < relation_postion[0] and relation_postion[1]<text_position[1]:
                    relation_argments = []
                    for argument in relation_mention['argments']:
                        try:
                            entity_type = entity_map[argument['entity-id']]['entity-type']
                        except KeyError:
                            print('[Warning] The entity in the other sentence is mentioned. This argument will be ignored.')
                            continue
                        relation_argments.append({
                            'entity-id': argument['entity-id'],
                            'entity-type': entity_type,
                            'position': argument['position'],
                            'role': argument['role'],
                            'text': self.clean_text(argument['text']),
                        })
                    item['golden-relation-mentions'].append({
                        'relation-type': relation_mention['relation-type'],
                        'text': self.clean_text(relation_mention['text']),
                        'position': relation_mention['position'],
                        "relation-id": relation_mention['relation-id'],
                        'arguments': relation_argments
                    })
            data.append(item)
        return data

    def find_correct_offset(self, sgm_text, start_index, text):
        offset = 0
        for i in range(0, 70):
            for j in [-1, 1]:
                offset = i * j
                if sgm_text[start_index + offset:start_index + offset + len(text)] == text:
                    return offset

        print('[Warning] fail to find offset! (start_index: {}, text: {}, path: {})'.format(start_index, text, self.sgm_path))
        return offset

    def fix_wrong_position(self):
        for entity_mention in self.entity_mentions:
            offset = self.find_correct_offset(
                sgm_text=self.sgm_text,
                start_index=entity_mention['position'][0],
                text=entity_mention['text'])

            entity_mention['position'][0] += offset
            entity_mention['position'][1] += offset
            entity_mention['head']["position"][0] += offset
            entity_mention['head']["position"][1] += offset

        for relation_mention in self.relation_mentions:
            offset1 = self.find_correct_offset(
                sgm_text=self.sgm_text,
                start_index=relation_mention['position'][0],
                text=relation_mention['text']
            )
            relation_mention['position'][0] += offset1
            relation_mention['position'][1] += offset1

            for argment in relation_mention['argments']:
                offset2 = self.find_correct_offset(
                    sgm_text=self.sgm_text,
                    start_index=argment['position'][0],
                    text=argment['text']
                )
                argment['position'][0] += offset2
                argment['position'][1] += offset2

        for event_mention in self.event_mentions:
            offset1 = self.find_correct_offset(
                sgm_text=self.sgm_text,
                start_index=event_mention['trigger']['position'][0],
                text=event_mention['trigger']['text'])
            event_mention['trigger']['position'][0] += offset1
            event_mention['trigger']['position'][1] += offset1

            for argument in event_mention['arguments']:
                offset2 = self.find_correct_offset(
                    sgm_text=self.sgm_text,
                    start_index=argument['position'][0],
                    text=argument['text'])
                argument['position'][0] += offset2
                argument['position'][1] += offset2

    def parse_sgm(self, sgm_path):
        with open(sgm_path, 'r') as f:
            soup = BeautifulSoup(f.read(), features='html.parser')
            self.sgm_text = soup.text

            doc_type = soup.doc.doctype.text.strip()

            def remove_tags(selector):
                tags = soup.findAll(selector)
                for tag in tags:
                    tag.extract()

            if doc_type == 'WEB TEXT':
                remove_tags('poster')
                remove_tags('postdate')
                remove_tags('subject')
            elif doc_type in ['CONVERSATION', 'STORY']:
                remove_tags('speaker')

            sents = []
            converted_text = soup.text

            for sent in nltk.sent_tokenize(converted_text):
                sents.extend(sent.split('\n\n'))
            sents = list(filter(lambda x: len(x) > 5, sents))
            sents = sents[1:]
            sents_with_pos = []
            last_pos = 0
            for sent in sents:
                pos = self.sgm_text.find(sent, last_pos)
                last_pos = pos
                sents_with_pos.append({
                    'text': sent,
                    'position': [pos, pos + len(sent)]
                })

            return sents_with_pos

    def parse_xml(self, xml_path):
        entity_mentions, event_mentions, relation_mentions = [], [], []
        tree = ET.parse(xml_path)
        root = tree.getroot()

        for child in root[0]:
            if child.tag == 'entity':
                entity_mentions.extend(self.parse_entity_tag(child))
            elif child.tag in ['value', 'timex2']:
                entity_mentions.extend(self.parse_value_timex_tag(child))
            elif child.tag == 'event':
                event_mentions.extend(self.parse_event_tag(child))
            elif child.tag == 'relation':
                relation_mentions.extend(self.parse_relation_tag(child))

        return entity_mentions, event_mentions, relation_mentions

    @staticmethod
    def parse_relation_tag(node):
        relation_mentions = []
        for child in node:
            if child.tag != 'relation_mention':
                continue
            relation_mention = dict()
            arguments = []
            for child2 in child:
                if child2.tag == 'relation_mention_argument':
                    extent = child2[0]
                    charset = extent[0]
                    arguments.append({
                        'text': charset.text,
                        'position': [int(charset.attrib['START']), int(charset.attrib['END'])],
                        'role': child2.attrib['ROLE'],
                        'entity-id': child2.attrib['REFID']
                    })
                elif child2.tag == 'extent':
                    charset = child2[0]
                    relation_mention['relation-id'] = node.attrib['ID']
                    relation_mention['relation-type'] = '{}:{}'.format(node.attrib['TYPE'], node.attrib['SUBTYPE'])
                    relation_mention['text'] = charset.text
                    relation_mention['position'] = [int(charset.attrib['START']), int(charset.attrib['END'])]

            relation_mention['argments'] = arguments
            relation_mentions.append(relation_mention)
        return relation_mentions

    @staticmethod
    def parse_entity_tag(node):
        entity_mentions = []

        for child in node:
            if child.tag != 'entity_mention':
                continue
            extent = child[0]
            head = child[1]
            charset = extent[0]
            head_charset = head[0]

            entity_mention = dict()
            entity_mention['entity-id'] = child.attrib['ID']
            entity_mention['entity-type'] = '{}:{}'.format(node.attrib['TYPE'], node.attrib['SUBTYPE'])
            entity_mention['text'] = charset.text
            entity_mention['position'] = [int(charset.attrib['START']), int(charset.attrib['END'])]
            entity_mention["head"] = {"text": head_charset.text,
                                      "position": [int(head_charset.attrib['START']), int(head_charset.attrib['END'])]}
            entity_mentions.append(entity_mention)

        return entity_mentions

    @staticmethod
    def parse_event_tag(node):
        event_mentions = []
        for child in node:
            if child.tag == 'event_mention':
                event_mention = dict()
                event_mention['event_type'] = '{}:{}'.format(node.attrib['TYPE'], node.attrib['SUBTYPE'])
                event_mention['arguments'] = []
                for child2 in child:
                    if child2.tag == 'ldc_scope':
                        charset = child2[0]
                        event_mention['text'] = charset.text
                        event_mention['position'] = [int(charset.attrib['START']), int(charset.attrib['END'])]
                    if child2.tag == 'anchor':   #anchor is trigger
                        charset = child2[0]
                        event_mention['trigger'] = {
                            'text': charset.text,
                            'position': [int(charset.attrib['START']), int(charset.attrib['END'])],
                        }
                    if child2.tag == 'event_mention_argument':
                        extent = child2[0]
                        charset = extent[0]
                        event_mention['arguments'].append({
                            'text': charset.text,
                            'position': [int(charset.attrib['START']), int(charset.attrib['END'])],
                            'role': child2.attrib['ROLE'],
                            'entity-id': child2.attrib['REFID'],
                        })
                event_mentions.append(event_mention)
        return event_mentions

    @staticmethod
    def parse_value_timex_tag(node):
        entity_mentions = []

        for child in node:
            extent = child[0]
            charset = extent[0]

            entity_mention = dict()
            entity_mention['entity-id'] = child.attrib['ID']

            if 'TYPE' in node.attrib:
                entity_mention['entity-type'] = node.attrib['TYPE']
            if 'SUBTYPE' in node.attrib:
                entity_mention['entity-type'] += ':{}'.format(node.attrib['SUBTYPE'])
            if child.tag == 'timex2_mention':
                entity_mention['entity-type'] = 'TIM:time'

            entity_mention['text'] = charset.text
            entity_mention['position'] = [int(charset.attrib['START']), int(charset.attrib['END'])]

            entity_mention["head"] = {"text": charset.text,
                                      "position": [int(charset.attrib['START']), int(charset.attrib['END'])]}

            entity_mentions.append(entity_mention)

        return entity_mentions

def find_token_index(tokens, start_pos, end_pos, phrase):
    start_idx, end_idx = -1, -1
    for idx, token in enumerate(tokens):
        if token['characterOffsetBegin'] <= start_pos:
            start_idx = idx

    assert start_idx != -1, "start_idx: {}, start_pos: {}, phrase: {}, tokens: {}".format(start_idx, start_pos, phrase, tokens)
    chars = ''

    def remove_punc(s):
        s = re.sub(r'[^\w]', '', s)
        return s

    for i in range(0, len(tokens) - start_idx):
        chars += remove_punc(tokens[start_idx + i]['originalText'])
        if remove_punc(phrase) in chars:
            end_idx = start_idx + i + 1
            break

    assert end_idx != -1, "end_idx: {}, end_pos: {}, phrase: {}, tokens: {}, chars:{}".format(end_idx, end_pos, phrase, tokens, chars)
    return start_idx, end_idx

class ACEDataset(Dataset):
    """
        This is dataset class for reading ACE dataset
    """
    def __init__(self, cfg: dict):
        super().__init__()
        _, self.root_path,  self.lang, self.mode, self.train_subset, self.dev_subset,self.test_subset, self.bert_tokenizer_path, self.stat_path  = cfg.values()
        self.tokenizer = get_tokenizer(self.bert_tokenizer_path)
        assert self.mode in ["train", "val", "test"]
        assert self.lang in ['English', 'Chinese', 'Arabic']
        self.all_train_sgm_files = []
        self.all_dev_sgm_files = []
        self.all_test_sgm_files = []
        self.all_train_base_names, self.all_dev_base_names, self.all_test_base_names = [], [], []
        for dir in self.train_subset:
            self.all_train_sgm_files += get_files_from_dir(os.path.join(self.root_path, self.lang, dir, 'adj'), '.sgm')
        for dir in self.dev_subset:
            self.all_dev_sgm_files += get_files_from_dir(os.path.join(self.root_path, self.lang, dir, 'adj'), '.sgm')
        for dir in self.test_subset:
            self.all_test_sgm_files += get_files_from_dir(os.path.join(self.root_path, self.lang, dir, 'adj'), '.sgm')

        for f in self.all_train_sgm_files:
            self.all_train_base_names.append(get_base_name(f, '.sgm'))
        for f in self.all_dev_sgm_files:
            self.all_dev_base_names.append(get_base_name(f, '.sgm'))
        for f in self.all_test_sgm_files:
            self.all_test_base_names.append(get_base_name(f, '.sgm'))

        self.data_subset_map = {
            "train": self.all_train_base_names,
            "val": self.all_dev_base_names,
            "test": self.all_test_base_names
        }
        if os.path.exists(self.stat_path):
            self.entity_types, self.event_types, self.relation_types = self.load_stat()
        else:
            self.entity_types, self.event_types, self.relation_types = self.stat()
        self.entity_types_num = len(self.entity_types)
        self.event_types_num = len(self.event_types)
        self.relation_types_num = len(self.relation_types)


    def stat(self):
        event_types, entity_types, relation_types = {}, {}, {}
        all_files = self.all_train_sgm_files + self.all_dev_sgm_files + self.all_test_sgm_files
        pbar = tqdm(all_files, desc="stating")
        for sgm_file in pbar:
            apf_file = self.get_apf_filename(sgm_file)
            annotations = self.preprocessing(sgm_path=sgm_file, apf_path=apf_file)
            for component in annotations['components']:
                for event in component['event']:
                    type = event['type']
                    if type not in event_types:
                        event_types[type] = len(event_types)
                for entity in component['entity']:
                    type = entity['type']
                    if type not in entity_types:
                        entity_types[type] = len(entity_types)

                for relation in component['relation']:
                    type = relation['type']
                    if type not in relation_types:
                        relation_types[type] = len(relation_types)
        with open(self.stat_path, 'w') as f:
            types_obj = {
                "entity-type": entity_types,
                "event-type": event_types,
                "relation-type": relation_types
            }
            json.dump(types_obj, f)

        return entity_types, event_types, relation_types

    def load_stat(self):
        with open(self.stat_path, 'r') as f:
            types_obj = json.load(f)
            entity_types, event_types, relation_types = types_obj["entity-type"], types_obj["event-type"], types_obj["relation-type"]
        return entity_types, event_types, relation_types

    def __getitem__(self, index):
        sgm_file = self.all_train_sgm_files[index]
        apf_file = self.get_apf_filename(sgm_file)
        annotations = self.preprocessing(sgm_path=sgm_file, apf_path=apf_file)
        triggers, event_ids = [],[]
        for component in annotations['components']:
            for event in component['event']:
                text, position, type = event['trigger'], event['trigger-position'], event['type']
                triggers.append({"tri": text, "pos": position, "type": type})
                event_ids.append(self.event_types[type])
        contexts = annotations['context']
        inputs = self.tokenize(contexts)
        atten_mask, word_mask1d, word_mask2d = self.get_masks(inputs)
        tri_targets = self.get_targets(inputs, triggers)
        pos_event_list, neg_event_list = self.get_event_list(event_ids)
        return (torch.LongTensor(inputs),
                torch.from_numpy(atten_mask),
                torch.from_numpy(word_mask1d),
                torch.from_numpy(word_mask2d),
                torch.from_numpy(tri_targets),
                torch.LongTensor(pos_event_list),
                torch.LongTensor(neg_event_list))

    def get_event_list(self, event_ids):
        total_event_set = set([i for i in range(self.event_types_num)])
        event_set = set()
        event_ids.append(-1) if len(event_ids) == 0 else None
        for i in event_ids:
            event_set.add(i)
        pos = list(event_set)
        neg = list(total_event_set - event_set)
        return pos, neg

    def tokenize(self, text):
        return [self.tokenizer.cls_token_id] + self.tokenizer.convert_tokens_to_ids([x for x in text.lower()]) + [self.tokenizer.sep_token_id]

    def get_apf_filename(self, sgm_file):
        return sgm_file.replace('.sgm', '.apf.xml')

    def get_ag_filename(self, sgm_file):
        return sgm_file.replace('.sgm', '.ag.xml')

    def get_masks(self, inputs):
        """
            获取MASK，在getitem阶段为1，全显示
            :param inputs: 模型输入的索引
            :return:
        """
        length = len(inputs) - 2
        _attn_mask = np.array([1] * len(inputs))
        _word_mask1d = np.array([1] * length)  # length: 语句长度=> _word_mask1d： 全为1可见， 去除start和end标记
        _word_mask2d = np.triu(np.ones((length, length), dtype=np.bool_))
        return _attn_mask, _word_mask1d, _word_mask2d

    def get_targets(self, inputs, triggers):
        length = len(inputs) - 2
        _event_labels = np.zeros((self.event_types_num, length, length), dtype=np.bool_)
        for tri in triggers:
            tri_type_id = self.event_types[tri['type']]
            _event_labels[tri_type_id, tri['pos'][0], tri['pos'][1]] = 1
        return _event_labels

    @staticmethod
    def collate_fn(batch):
        inputs, atten_mask, word_mask1d, word_mask2d, tri_targets, pos_events, neg_events = zip(*batch)
        max_tokens = np.max([x.shape[0] for x in word_mask1d])

        bs = len(inputs)
        inputs = pad_sequence(inputs, True)  # 长度按照最大的算，以0进行填充
        atten_mask = pad_sequence(atten_mask, True)  # 长度按照最大的算，以0进行填充
        word_mask1d = pad_sequence(word_mask1d, True)  # 长度按照最大的算，以0进行填充
        pos_events = pad_sequence(pos_events, True, -1)
        neg_events = pad_sequence(neg_events, True, -1)

        word_mask2d = pad_2d(word_mask2d, (bs, max_tokens, max_tokens))
        tri_targets = pad_3d(tri_targets, (bs, tri_targets[0].shape[0], max_tokens, max_tokens))

        return inputs, atten_mask, word_mask1d, word_mask2d, tri_targets, pos_events, neg_events

    def __len__(self):
        return self.get_length()

    def preprocessing(self, sgm_path, apf_path):
        parser = Parser(sgm_path=sgm_path, apf_path=apf_path)
        _context = parser.sgm_text
        args = []
        for i, item in enumerate(parser.get_data()):
            _arg = {}
            _arg['sentence'] = item['sentence']
            _arg['position'] = item['position']
            entitys, events, relations = [], [], []
            for j, entity in enumerate(item['golden-entity-mentions']):
                _entity = {
                    'id': entity['entity_id'],
                    'text': entity['text'],
                    'type': entity['entity-type'],
                    'position': entity['position'],
                    'head': entity['head']
                }
                entitys.append(_entity)
            for j, event in enumerate(item['golden-event-mentions']):
                _event = {
                    'trigger': event['trigger']['text'],
                    'trigger-position': event['trigger']['position'],
                    'arguments': event['arguments'],
                    'type': event['event_type'],
                    'position': event['position']
                }
                events.append(_event)
            for j, relation in enumerate(item['golden-relation-mentions']):
                _relation = {
                    'id': relation['relation-id'],
                    'type': relation['relation-type'],
                    'text': relation['text'],
                    'position': relation['position'],
                    'arguments': relation['arguments']
                }
                relations.append(_relation)
            _arg['event'] = events
            _arg['entity'] = entitys
            _arg['relation'] = relations
            args.append(_arg)
        annotation = {
           'context': _context,
            'components': args
        }
        return annotation

    def get_length(self):
        return len(self.data_subset_map[self.mode])






