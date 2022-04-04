



# def create_dataloader(path, imgsz, batch_size, stride, cls_map, single_cls, hyp=None, augment=False, cache=False, pad=0.0, rect=False,
#                       rank=-1, world_size=1, workers=8, image_weights=False,version='v5'):
def create_dataloader(path, imgsz, batch_size, stride, opt, augment=False, rank=-1, pad=0.0, rect=False):
    # Make sure only the first process in DDP process the dataset first, and the following others can use the cache
    with torch_distributed_zero_first(rank):
        dataset = LoadImagesAndLabels_MMDET(path, imgsz, batch_size,
                                      augment=augment,  # augment images
                                      hyp=opt.hyp,  # augmentation hyperparameters
                                      rect=opt.train_cfg.get('rect',None),  # rectangular training
                                      cache_images=opt.train_cfg.get('cache_images',None),
                                      cls_map = opt.data['cls_map'],
                                      single_cls=opt.train_cfg.get('single_cls',False),
                                      stride=int(stride) if isinstance(stride,int) else int(max(stride)),
                                      pad=pad,
                                      rank=rank,
                                      image_weights=opt.train_cfg.get('image_weights',None),
                                      version=opt.train_cfg['version'],
                                      debug = opt.train_cfg['debug'])                                      
        



    workers=opt.train_cfg['workers']
    image_weights=opt.train_cfg.get('image_weights',None)
    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count() // opt.world_size, batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if rank != -1 else None
    loader = torch.utils.data.DataLoader if image_weights else InfiniteDataLoader
    # Use torch.utils.data.DataLoader() if dataset.properties will update during training else InfiniteDataLoader()
    dataloader = loader(dataset,
                        shuffle=sampler is None,
                        batch_size=batch_size,
                        num_workers=nw,
                        sampler=sampler,
                        pin_memory=True,
                        collate_fn=LoadImagesAndLabels_MMDET.collate_fn)
    return dataloader, dataset