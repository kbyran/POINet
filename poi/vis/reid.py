import cv2
import logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mxnet as mx
from poi.ops.fuse.metric_ops import _c_distance


def vis_reid(image_url_a, feature_a, image_url_b, feature_b):
    feature_a_nd = mx.nd.L2Normalization(mx.nd.array([feature_a]), mode="instance")
    feature_b_nd = mx.nd.L2Normalization(mx.nd.array([feature_b]), mode="instance")
    score = 1. - _c_distance(mx.nd, feature_a_nd, feature_b_nd).asnumpy() / 2.
    log = "Cosine similarity: {}".format(score[0, 0])
    logging.info(log)

    # subplots with 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(log)

    # column 1
    img_a = cv2.imread(image_url_a)
    img_a = cv2.cvtColor(img_a, cv2.COLOR_RGB2BGR)
    img_a = cv2.resize(img_a, (256, 512), interpolation=cv2.INTER_LINEAR)
    ax1.imshow(img_a)
    ax1.axis("off")

    # column 2
    img_b = cv2.imread(image_url_b)
    img_b = cv2.cvtColor(img_b, cv2.COLOR_RGB2BGR)
    img_b = cv2.resize(img_b, (256, 512), interpolation=cv2.INTER_LINEAR)
    ax2.imshow(img_b)
    ax2.axis("off")

    plt.subplots_adjust(top=0.9, bottom=0.1, right=0.98, left=0.02, hspace=0, wspace=0.05)
    plt.margins(0, 0)

    save_url = image_url_a.split(".")[0] + "_reid.jpg"
    plt.savefig(save_url)
    logging.info("Visualization in {}".format(save_url))
