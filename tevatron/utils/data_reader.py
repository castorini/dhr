import logging
from tqdm import tqdm
import os
import json
from typing import List, Tuple
from collections import defaultdict
logger = logging.getLogger(__name__)

def create_dir(dir_: str):
    output_parent = '/'.join((dir_).split('/')[:-1])
    if not os.path.exists(output_parent):
        logger.info(f'Create {output_parent}')
        os.mkdir(output_parent)
    if not os.path.exists(dir_):
        logger.info(f'Create {dir_}')
        os.mkdir(dir_)

def read_tsv(path: str):
    id2info = {}
    with open(path, 'r') as f:
        for line in tqdm(f, desc=f"read {path}"):
            idx, info = line.strip().split('\t')
            id2info[idx] = info
    return id2info

def read_json(path: str,
              id_key: str = 'id',
              content_key: str = 'content',
              meta_keys: List[str] = None,
              sep: str = ' '):
    id2info = {}
    with open(path, 'r') as f:
        for line in tqdm(f, desc=f"read {path}"):
            data = json.loads(line.strip().split('\t'))
            idx = data[id_key]
            info = data[content_key]
            if meta_key:
                info = [info]
                for meta_key in meta_keys:
                    info.append(data[meta_key])
                info = sep.join(info)
            id2info[idx] = info
    return id2info

def read_trec(path: str):
    qid2psg = defaultdict(list)
    with open(path, 'r') as f:
        for line in tqdm(f, desc=f"read {path}"):
            try:
                data = line.strip().split('\t')
                qid = data[0]
                psg = data[2]
            except:
                data = line.strip().split(' ')
                qid = data[0]
                psg = data[2]
            qid2psg[qid].append(psg)
            

    return qid2psg

def read_qrel(path: str):
    qid_pid2qrel = defaultdict(int)
    with open(path, 'r') as f:
        for line in tqdm(f, desc=f"read {path}"):
            qid, _, pid, rel,= line.strip().split('\t')
            qid_pid2qrel[f'{qid}_{pid}'] = int(rel)
    return qid_pid2qrel