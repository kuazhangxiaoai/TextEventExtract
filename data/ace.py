import os
import json
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import nltk
import re
from data.utils import get_files_from_dir, get_base_name, get_pipline, get_tokenizer, find_word_positions
from torch.utils.data import Dataset

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
                item['golden-relation-mentions'].append({
                    'relation-type': relation_mention['relation-type'],
                    'text': self.clean_text(relation_mention['text']),
                    'position': relation_mention['position'],
                    "relation-id": relation_mention['relation-id']
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

            for argment in relation_mention['arguments']:
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
                    if child2.tag == 'anchor':
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
        _, self.root_path,  self.lang, self.mode, self.train_subset, self.dev_subset,self.test_subset, self.nlp_tools, self.bert_tokenizer_path  = cfg.values()
        self.pipline = get_pipline(weight_dir=self.nlp_tools, lang=self.lang)
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

    def __getitem__(self, index):
        sgm_file = self.all_train_sgm_files[index]
        apf_file = self.get_apf_filename(sgm_file)

        ace_list = self.preprocessing(sgm_path=sgm_file, apf_path=apf_file)
        _inputs = []
        for i, item in enumerate(ace_list):
            _input = {}
            sentence = item['sentence']
            _words = self.tokenizer.tokenize(sentence)
            _index = self.tokenizer(sentence)
            _input['words'] = _words
            _input['index'] = _index.data['input_ids']
            _entites, _events = [],[]
            for i, entity_mention in enumerate(item['golden-entity-mentions']):
                _entity = {
                    'text': entity_mention['text'],
                    'type': entity_mention['entity-type'],
                    'position': entity_mention['position'],
                    'position-sentence': find_word_positions(sentence, entity_mention['text'])[0],
                }
                _entites.append(_entity)

            for i, event_mention in enumerate(item['golden-event-mentions']):
                _event = {

                }
                _events.append(_event)

            _input['entites'] = _entites
            _input['event'] = _events
            _inputs.append(_input)

        return _inputs

    def get_apf_filename(self, sgm_file):
        return sgm_file.replace('.sgm', '.apf.xml')

    def get_ag_filename(self, sgm_file):
        return sgm_file.replace('.sgm', '.ag.xml')


    @staticmethod
    def collate_fn(batch):
        return None

    def __len__(self):
        return self.get_length()

    def preprocessing(self, sgm_path, apf_path):
        parser = Parser(sgm_path=sgm_path, apf_path=apf_path)
        for i, item in enumerate(parser.get_data()):
            _input = {}
            sentence = item['sentence']
            _words = self.tokenizer.tokenize(sentence)
            _index = self.tokenizer(sentence)
            _input['words'] = _words
            _input['index'] = _index.data['input_ids']
            gold_entity_mentions, gold_event_mentions = [],[]
            #_inputs.append(_input)
        return parser.get_data()

    def get_length(self):
        return len(self.data_subset_map[self.mode])




