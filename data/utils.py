import os
import stanza
from torch.utils.data import DataLoader

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



def get_triggers(argment:list[object]):
    return [arg['trigger'] for arg in argment]


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


