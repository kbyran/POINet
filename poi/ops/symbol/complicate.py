from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import mxnet as mx


__all__ = ["normalizer_factory", "bn_count"]

bn_count = [0]


def normalizer_factory(type="local", ndev=None, eps=1e-5 + 1e-10, mom=0.9, lr_mult=1.0,
                       wd_mult=1.0):
    """
    :param type: one of "fix", "local", "sync"
    :param ndev:
    :param eps:
    :param mom: momentum of moving mean and moving variance
    :return: a wrapper with signature, bn(data, name)
    """
    # sometimes the normalizer may be pre-constructed
    if callable(type):
        return type

    if type == "local" or type == "localbn":
        def local_bn(data, gamma=None, beta=None, moving_var=None, moving_mean=None,
                     name=None, momentum=mom, lr_mult=lr_mult, wd_mult=wd_mult):
            if name is None:
                prev_name = data.name
                name = prev_name + "_bn"
            return mx.sym.BatchNorm(data=data,
                                    gamma=gamma,
                                    beta=beta,
                                    moving_var=moving_var,
                                    moving_mean=moving_mean,
                                    name=name,
                                    fix_gamma=False,
                                    use_global_stats=False,
                                    momentum=momentum,
                                    eps=eps,
                                    lr_mult=lr_mult,
                                    wd_mult=wd_mult)
        return local_bn

    elif type == "fix" or type == "fixbn":
        def fix_bn(data, gamma=None, beta=None, moving_var=None, moving_mean=None,
                   name=None, lr_mult=lr_mult, wd_mult=wd_mult):
            if name is None:
                prev_name = data.name
                name = prev_name + "_bn"
            return mx.sym.BatchNorm(data=data,
                                    gamma=gamma,
                                    beta=beta,
                                    moving_var=moving_var,
                                    moving_mean=moving_mean,
                                    name=name,
                                    fix_gamma=False,
                                    use_global_stats=True,
                                    eps=eps,
                                    lr_mult=lr_mult,
                                    wd_mult=wd_mult)
        return fix_bn

    elif type == "sync" or type == "syncbn":
        assert ndev is not None, "Specify ndev for sync bn"

        def sync_bn(data, gamma=None, beta=None, moving_var=None, moving_mean=None,
                    name=None, momentum=mom, lr_mult=lr_mult, wd_mult=wd_mult):
            bn_count[0] = bn_count[0] + 1
            if name is None:
                prev_name = data.name
                name = prev_name + "_bn"
            return mx.sym.contrib.SyncBatchNorm(data=data,
                                                gamma=gamma,
                                                beta=beta,
                                                moving_var=moving_var,
                                                moving_mean=moving_mean,
                                                name=name,
                                                fix_gamma=False,
                                                use_global_stats=False,
                                                momentum=momentum,
                                                eps=eps,
                                                ndev=ndev,
                                                key=str(bn_count[0]),
                                                lr_mult=lr_mult,
                                                wd_mult=wd_mult)
        return sync_bn

    elif type == "in":
        def in_(data, gamma=None, beta=None,
                name=None, lr_mult=lr_mult, wd_mult=wd_mult):
            if name is None:
                prev_name = data.name
                name = prev_name + "_in"
            name = name.replace("_bn", "_in")
            return mx.sym.InstanceNorm(data=data,
                                       gamma=gamma,
                                       beta=beta,
                                       name=name,
                                       eps=eps,
                                       lr_mult=lr_mult,
                                       wd_mult=wd_mult)
        return in_

    elif type == "gn":
        def gn(data, gamma=None, beta=None,
               name=None, lr_mult=lr_mult, wd_mult=wd_mult):
            if name is None:
                prev_name = data.name
                name = prev_name + "_gn"
            name = name.replace("_bn", "_gn")
            return mx.sym.contrib.GroupNorm(data=data,
                                            gamma=gamma,
                                            beta=beta,
                                            name=name,
                                            eps=eps,
                                            num_group=32,
                                            lr_mult=lr_mult,
                                            wd_mult=wd_mult)
        return gn

    elif type == "ibn":
        def ibn(data, in_gamma=None, in_beta=None, bn_gamma=None, bn_beta=None, bn_moving_var=None,
                bn_moving_mean=None, name=None, lr_mult=lr_mult, wd_mult=wd_mult):
            if name is None:
                prev_name = data.name
                name = prev_name + "_ibn"
            name = name.replace("_bn", "_ibn")
            split = mx.sym.split(data=data, num_outputs=2, name=name + "_split")
            out1 = mx.sym.InstanceNorm(data=split[0],
                                       gamma=in_gamma,
                                       beta=in_beta,
                                       name=name + "_in",
                                       eps=eps,
                                       lr_mult=lr_mult,
                                       wd_mult=wd_mult)
            out2 = mx.sym.BatchNorm(data=split[1],
                                    gamma=bn_gamma,
                                    beta=bn_beta,
                                    moving_var=bn_moving_var,
                                    moving_mean=bn_moving_mean,
                                    name=name + "_bn",
                                    fix_gamma=False,
                                    use_global_stats=False,
                                    eps=eps,
                                    lr_mult=lr_mult,
                                    wd_mult=wd_mult)
            return mx.sym.Concat(out1, out2, dim=1, name=name + "concat")

        return ibn

    elif type == "dummy":
        def dummy(data, name=None):
            return data
        return dummy
    else:
        raise KeyError("Unknown norm type {}".format(type))
