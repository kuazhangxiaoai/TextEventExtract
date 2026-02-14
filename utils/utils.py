import yaml

def load_config(config_file):
    with open(config_file) as f:
        cfg = yaml.safe_load(f)
    return cfg

def make_config(data_cfg_path, model_cfg_path, hyp_cfg_path):
    data_cfg = load_config(data_cfg_path)
    model_cfg = load_config(model_cfg_path)
    hyp_cfg = load_config(hyp_cfg_path)
    cfg = {
        'data': data_cfg,
        'model': model_cfg,
        'hyp': hyp_cfg
    }
    return cfg

def split_tensor_sliding_window(input_ids, attention_mask, max_len=512, stride=512):
    """
    input_ids: [seq_len]
    attention_mask: [seq_len]
    return: List of (input_ids_chunk, attention_mask_chunk)
    """
    chunks_ids,chunks_attns = [],[]
    start = 0
    seq_len = input_ids.size(1)

    while start < seq_len:
        end = start + max_len
        chunk_ids = input_ids[:, start:end]
        chunk_mask = attention_mask[:, start:end]
        chunks_ids.append(chunk_ids)
        chunks_attns.append(chunk_mask)
        start += stride

    return chunks_ids, chunks_attns
