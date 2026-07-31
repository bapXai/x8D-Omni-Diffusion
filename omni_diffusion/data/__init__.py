# data.__init__ — lazy: byte-native processor imports must not drag in torch.

def build_supervised_dataset_deepspeed(*args, **kwargs):
    from .build import build_supervised_dataset_deepspeed as _impl
    return _impl(*args, **kwargs)
