import cv2
import logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def vis_pose(image_url, affine, max_idx, max_val, kp_names, skeletons):
    # subplots with 3 columns
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.set_size_inches(8, 6)
    fig.suptitle("Keypoints for {}".format(image_url), fontsize="x-large")

    # column 1
    img = cv2.imread(image_url)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img1 = cv2.resize(img, (256, 512), interpolation=cv2.INTER_LINEAR)
    ax1.imshow(img1)
    ax1.axis("off")

    # dotted image for column 2 and 3
    inv_affine = np.linalg.inv(affine)
    affine_pts = np.ones((len(max_idx), 3))
    affine_pts[:, :2] = np.array(max_idx) * 4.0
    pts = np.dot(inv_affine, affine_pts.T).T[:, :2]
    img_dotted = (0.5 * img).astype(np.uint8)

    # column 2
    img2 = img_dotted.copy()
    for p, name in zip(pts, kp_names):
        cv2.circle(img2, (int(p[0]), int(p[1])), 8, (0, 128, 255),
                   thickness=-1, lineType=cv2.FILLED)
        cv2.putText(img2, "{}".format(name), (int(p[0]) + 10, int(p[1]) + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 128), 1, lineType=cv2.LINE_AA)
    img2 = cv2.resize(img2, (256, 512), interpolation=cv2.INTER_LINEAR)
    ax2.imshow(img2)
    ax2.axis("off")

    # column 3
    img3 = img_dotted.copy()
    for p in pts:
        cv2.circle(img3, (int(p[0]), int(p[1])), 8, (0, 128, 255),
                   thickness=-1, lineType=cv2.FILLED)
    for (i, j) in skeletons:
        pt_a = pts[i]
        pt_b = pts[j]
        cv2.line(img3, (int(pt_a[0]), int(pt_a[1])),
                 (int(pt_b[0]), int(pt_b[1])), (255, 255, 128), 2)
    img3 = cv2.resize(img3, (256, 512), interpolation=cv2.INTER_LINEAR)
    ax3.imshow(img3)
    ax3.axis("off")

    # adjust margins
    plt.subplots_adjust(top=1, bottom=0.01, right=0.98, left=0.02, hspace=0, wspace=0.05)
    plt.margins(0, 0)

    # save figure
    save_url = image_url.split(".")[0] + "_pose.jpg"
    plt.savefig(save_url)
    logging.info("Visualization in {}".format(save_url))
