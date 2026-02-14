from model.network import OneEventExtracter, TriNet

dataset_map = {
    "triNet": TriNet,
    "oneEE": OneEventExtracter
}

__all__ = [
    "OneEventExtracter",
    "TriNet"
]