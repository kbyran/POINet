import cv2
import logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def vis_densepose(image_url, gt_bbox, ann_index_lowres, index_uv_lowres, u_lowres, v_lowres):
    # preprocess
    gt_bbox = [int(v) for v in gt_bbox]
    x1, y1, x2, y2 = gt_bbox
    width = x2 - x1
    height = y2 - y1

    # subplots with 5 columns
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5)
    fig.set_size_inches(11, 5)
    fig.suptitle("DensePose for {}".format(image_url), fontsize="x-large")

    # column 1 for raw image
    img = cv2.imread(image_url)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img1 = cv2.resize(img, (256, 512), interpolation=cv2.INTER_LINEAR)
    ax1.imshow(img1)
    ax1.axis("off")

    # column 1
    img2 = img.copy()
    ann_index_lowres = np.transpose(ann_index_lowres, (1, 2, 0))
    ann_index = cv2.resize(ann_index_lowres, (width, height), interpolation=cv2.INTER_LINEAR)
    ann_index = np.argmax(ann_index, axis=2)
    ann_index_mask = np.tile((ann_index == 0)[:, :, np.newaxis], [1, 1, 3])
    ann_index_vis = cv2.applyColorMap(
        (ann_index * 15).astype(np.uint8), cv2.COLORMAP_PARULA)[:, :, :]
    ann_index_vis[ann_index_mask] = img2[y1: y2, x1: x2, :][ann_index_mask]
    img2[y1: y2, x1: x2, :] = img2[y1: y2, x1: x2, :] * 0.3 + ann_index_vis * 0.7
    img2 = cv2.resize(img2, (256, 512), interpolation=cv2.INTER_LINEAR)
    ax2.imshow(img2)
    ax2.axis("off")

    # column 3
    img3 = np.zeros_like(img[:, :, 0]).astype(np.float32)
    index_uv_lowres = np.transpose(index_uv_lowres, (1, 2, 0))
    index_uv_lowres = cv2.resize(index_uv_lowres, (width, height), interpolation=cv2.INTER_LINEAR)
    index_uv_vis = np.argmax(index_uv_lowres, axis=2).astype(np.float32) / 24
    ann_index_mask = ann_index == 0
    index_uv_vis[ann_index_mask] = img3[y1: y2, x1: x2][ann_index_mask]
    img3[y1: y2, x1: x2] = img3[y1: y2, x1: x2] * 0.3 + index_uv_vis * 0.7
    img3 = cv2.resize(img3, (256, 512), interpolation=cv2.INTER_LINEAR)
    ax3.imshow(img3)
    ax3.axis("off")

    # column 4
    img4 = np.zeros_like(img[:, :, 0]).astype(np.float32)
    u_lowres = np.transpose(u_lowres, (1, 2, 0))
    u_lowres = cv2.resize(u_lowres, (width, height), interpolation=cv2.INTER_LINEAR)
    u_lowres = u_lowres[np.arange(height)[:, None], np.arange(
        width)[None, :], np.argmax(index_uv_lowres, axis=2)]
    ann_index_mask = ann_index == 0
    u_lowres[ann_index_mask] = img4[y1: y2, x1: x2][ann_index_mask]
    img4[y1: y2, x1: x2] = img4[y1: y2, x1: x2] * 0.3 + u_lowres * 0.7
    img4 = cv2.resize(img4, (256, 512), interpolation=cv2.INTER_LINEAR)
    ax4.imshow(img4)
    ax4.axis("off")

    # column 5
    img5 = np.zeros_like(img[:, :, 0]).astype(np.float32)
    v_lowres = np.transpose(v_lowres, (1, 2, 0))
    v_lowres = cv2.resize(v_lowres, (width, height), interpolation=cv2.INTER_LINEAR)
    v_lowres = v_lowres[np.arange(height)[:, None], np.arange(
        width)[None, :], np.argmax(index_uv_lowres, axis=2)]
    ann_index_mask = ann_index == 0
    v_lowres[ann_index_mask] = img5[y1: y2, x1: x2][ann_index_mask]
    img5[y1: y2, x1: x2] = img5[y1: y2, x1: x2] * 0.3 + v_lowres * 0.7
    img5 = cv2.resize(img5, (256, 512), interpolation=cv2.INTER_LINEAR)
    ax5.imshow(img5)
    ax5.axis("off")

    plt.subplots_adjust(top=1, bottom=0.01, right=0.98, left=0.02, hspace=0, wspace=0.05)
    plt.margins(0, 0)

    save_url = image_url.split(".")[0] + "_densepose.jpg"
    plt.savefig(save_url)
    logging.info("Visualization in {}".format(save_url))
