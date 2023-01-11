# Inheritaed from open-mmlab/mmclassification
# Modified by Hangyu LIN

def wrap_non_distributed_model(model, device='cuda', dim=0, *args, **kwargs):
    """Wrap module in non-distributed environment by device type.

    - For CUDA, wrap as to.device.
    
    - For CPU not wrap the model.

    Args:
        model(:class:`nn.Module`): model to be parallelized.
        device(str): device type, cuda, cpu or mlu. Defaults to cuda.
        dim(int): Dimension used to scatter the data. Defaults to 0.

    Returns:
        model(nn.Module): the model to be parallelized.
    """
    
    if 'cuda' in device:
        device_id = kwargs['device_ids'][0]
        model = model.to(f'cuda:{device_id}')
    elif device == 'cpu':
        model = model.cpu()
    else:
        raise RuntimeError(f'Unavailable device "{device}"')

    return model


def wrap_distributed_model(model, device='cuda', *args, **kwargs):
    """Build DistributedDataParallel module by device type.

    - Other device types are not supported by now.

    Args:
        model(:class:`nn.Module`): module to be parallelized.
        device(str): device type, mlu or cuda.

    Returns:
        model(:class:`nn.Module`): the module to be parallelized

    References:
        .. [1] https://pytorch.org/docs/stable/generated/torch.nn.parallel.
               DistributedDataParallel.html
    """
    
    if 'cuda' in device:
        from torch.cuda import current_device
        from torch.nn.parallel import DistributedDataParallel
        model = DistributedDataParallel(
            model.cuda(), *args, device_ids=[current_device()], **kwargs)
    else:
        raise RuntimeError(f'Unavailable device "{device}"')

    return model
