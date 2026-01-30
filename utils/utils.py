import yaml

def load_config(config_file):
    with open(config_file) as f:
        cfg = yaml.safe_load(f)
    return cfg
