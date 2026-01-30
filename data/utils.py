from torch.utils.data import DataLoader

def collate_fn(batch):
    t, e = batch

    return t, e

def build_dataloader(dataset, batchsize, workers):
    return DataLoader(
        dataset,
        batchsize,
        True,
        None,
        None,
        num_workers=workers,
        collate_fn=collate_fn
    )


