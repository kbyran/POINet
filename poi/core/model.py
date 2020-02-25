from mxnet import nd
import logging


def load_params(prefix, epoch):
    """Load params from a file
    """
    save_dict = nd.load("%s-%04d.params" % (prefix, epoch))
    arg_params = {}
    aux_params = {}
    if not save_dict:
        logging.warning('params file "%s" is empty', '%s-%04d.params' % (prefix, epoch))
        return (arg_params, aux_params)
    for k, v in save_dict.items():
        tp, name = k.split(":", 1)
        if tp == "arg":
            arg_params[name] = v
        if tp == "aux":
            aux_params[name] = v
    logging.info('params file "%s" is loaded', '%s-%04d.params' % (prefix, epoch))
    return (arg_params, aux_params)
