import os
import stanza
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

LANG_MAP = {
    "English": "en",
    "Chinese": "zh",
    "Arabic": "ar"
}

PIPLINE_PROTERTIE_MAP = {
    "English": 'tokenize,pos,lemma',
    "Chinese": 'tokenize,pos,ner',
    "Arabic": 'tokenize,pos,lemma',
}

def get_pipline(weight_dir: str,lang: str):
    pros = PIPLINE_PROTERTIE_MAP[lang]
    lang_s = LANG_MAP[lang]
    pipline = stanza.Pipeline(lang_s, processors=pros, dir=weight_dir, download_method=None, use_gpu=False)
    return pipline

def get_tokenizer(path):
    return AutoTokenizer.from_pretrained(path)

def get_triggers(argment:list[object]):
    return [arg['trigger'] for arg in argment]

def find_word_positions(text, word):
    """
    找出 word 在 text 中的所有位置（字符偏移，0-based）

    Args:
        text (str): 原文
        word (str): 要查找的词或短语

    Returns:
        List[Tuple[int, int]]: 每个匹配的 (start, end)，end 包含最后一个字符
    """
    positions = []
    start = 0
    while True:
        idx = text.find(word, start)
        if idx == -1:
            break
        positions.append((idx, idx + len(word) - 1))
        start = idx + 1  # 继续搜索下一个位置
    return positions

def build_dataloader(dataset, batchsize, workers, shuffle):
    return DataLoader(
        dataset,
        batchsize,
        shuffle,
        None,
        None,
        num_workers=workers,
        collate_fn=dataset.collate_fn
    )

def get_files_from_dir(dir,ext = None):
  allfiles = []
  needExtFilter = (ext != None)
  for root,dirs,files in os.walk(dir):
    for filespath in files:
      filepath = os.path.join(root, filespath)
      extension = os.path.splitext(filepath)[1][1:]
      if needExtFilter and extension in ext:
        allfiles.append(filepath)
      elif not needExtFilter:
        allfiles.append(filepath)
  return allfiles

def get_base_name(fullname: str, ext: str):
    return fullname.split(os.sep)[-1].replace(ext, '')

def pad_2d(data, dim, value=0):
    new_data = torch.zeros(dim, dtype=torch.bool) + value
    for j, x in enumerate(data):
        new_data[j, :x.shape[0], :x.shape[1]] = x
    return new_data

def pad_3d(data, dim, value=0):
    new_data = torch.zeros(dim, dtype=torch.bool) + value
    for j, x in enumerate(data):
        new_data[j, :, :x.shape[1], :x.shape[2]] = x
    return new_data

def pad_4d(data, dim, value=0):
    new_data = torch.zeros(dim, dtype=torch.bool) + value
    for j, x in enumerate(data):
        new_data[j, :, :x.shape[1], :x.shape[2], :] = x
    return new_data
