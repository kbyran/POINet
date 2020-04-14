import os
import mxnet as mx
from gluoncv.model_zoo.model_store import get_model_file

mxnet_gluon_repo = os.environ.get("MXNET_GLUON_REPO", None)
if not mxnet_gluon_repo:
    os.environ["MXNET_GLUON_REPO"] = "https://apache-mxnet.s3.cn-north-1.amazonaws.com.cn"

num_layers = 50
version = 1
pretrained = True
root = "./pretrained"
file_path = get_model_file('resnet%d_v%d' % (num_layers, version), tag=pretrained, root=root)
print("models is saved in {}".format(file_path))

gcv_params = mx.nd.load(file_path)

cvt_params = dict()
for k in gcv_params:
    if not k.startswith("features"):
        continue
    k_list = k.split(".")
    if k_list[1] == "0":
        cvt_k = "arg:conv0_" + k_list[2]
        cvt_params[cvt_k] = gcv_params[k]
    elif k_list[1] == "1":
        if k_list[-1].endswith("running_mean"):
            cvt_k = "aux:bn0_moving_mean"
        elif k_list[-1].endswith("running_var"):
            cvt_k = "aux:bn0_moving_var"
        else:
            cvt_k = "arg:bn0_" + k_list[2]
    else:
        stage = "stage{}".format(int(k_list[1]) - 3)
        unit = "unit{}".format(int(k_list[2]) + 1)
        if k_list[3] == "downsample":
            if k_list[4] == "0":
                layer = "sc"
            elif k_list[4] == "1":
                layer = "sc_bn"
        elif k_list[3] == "body":
            if k_list[4] == "0":
                layer = "conv1"
            elif k_list[4] == "1":
                layer = "bn1"
            elif k_list[4] == "3":
                layer = "conv2"
            elif k_list[4] == "4":
                layer = "bn2"
            elif k_list[4] == "6":
                layer = "conv3"
            elif k_list[4] == "7":
                layer = "bn3"
        if k_list[5].endswith("running_mean"):
            prefix = "aux"
            postfix = "moving_mean"
        elif k_list[5].endswith("running_var"):
            prefix = "aux"
            postfix = "moving_var"
        else:
            prefix = "arg"
            postfix = k_list[5]
        cvt_k = "{}:{}_{}_{}_{}".format(prefix, stage, unit, layer, postfix)
    print("{}-->{}".format(k, cvt_k))
    cvt_params[cvt_k] = gcv_params[k].copy()

new_file_path = "{}-0000.params".format(file_path.split("-")[0])
mx.nd.save(new_file_path, cvt_params)
