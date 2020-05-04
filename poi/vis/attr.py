import cv2
import pprint
import logging
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np


def vis_attr(image_url, attr_names, attr_outputs):
    # subplots with 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={"width_ratios": [1, 2]})
    fig.suptitle("The attributes for {}".format(image_url), fontsize="x-large")
    fig.set_size_inches(6, 3)

    # column 1
    img = cv2.imread(image_url)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (256, 512), interpolation=cv2.INTER_LINEAR)
    ax1.imshow(img)
    ax1.axis("off")

    # column 2
    attrs = sorted(zip(attr_names, attr_outputs), key=lambda x: x[1], reverse=True)
    logging.info(pprint.pformat(attrs))
    x_name = [d[0] for d in attrs[:5]][::-1]
    x_value = [d[1] for d in attrs[:5]][::-1]
    x = np.arange(len(x_name)) * 2
    rects = plt.barh(x, x_value, color="#1b71f1", align="center")
    ax2.xaxis.set_tick_params(labelsize=15)
    ax2.set_xticks([])
    ax2.set_yticks(x)
    ax2.set_yticklabels(x_name)
    ax2.set_xlim(0, 1.3)
    ax2.set_xlabel("Score")
    ax2.set_ylabel("Attribute")
    rect_labels = []
    for rect in rects:
        width = rect.get_width()
        rankStr = "{:.4f}".format(width)
        yloc = rect.get_y() + rect.get_height() / 2
        label = ax2.annotate(rankStr, xy=(width, yloc), xytext=(5, 0),
                             textcoords="offset points",
                             ha="left", va="center",
                             color="black", weight="bold", clip_on=True)
        rect_labels.append(label)
    for key, spine in ax2.spines.items():
        if key in ["left", "right", "bottom", "top"]:
            spine.set_visible(False)

    # adjust margins
    plt.subplots_adjust(top=0.9, bottom=0.1, right=0.98, left=0.02, hspace=0, wspace=1.0)
    plt.margins(0, 0)

    # save figure
    save_url = image_url.split(".")[0] + "_attr.jpg"
    plt.savefig(save_url)
    logging.info("Visualization in {}".format(save_url))
