import cv2
import numpy as np


class Augmenter(object):
    """Image Augmenter base class"""

    def __init__(self, **kwargs):
        raise NotImplementedError

    def __call__(self, **kwargs):
        raise NotImplementedError


class ReadRecord(Augmenter):
    """
    input: input_record
    output: input_record with specific items
    """

    def __init__(self, pRead):
        self.p = pRead  # type: ReadParam

    def __call__(self, input_record):
        p = self.p
        items = p.items
        for item in items:
            if item == "image":
                image = cv2.imread(input_record["image_url"], cv2.IMREAD_COLOR)
                input_record["image"] = image[:, :, ::-1].astype("float32")
            elif item == "label":
                input_record["label"] = np.array(input_record["pid"], dtype="float32")
            elif item == "labels":
                input_record["labels"] = np.array(input_record["labels"], dtype="float32")
            elif item == "gt_bbox":
                input_record["gt_bbox"] = np.concatenate(
                    [input_record["gt_bbox"], input_record["gt_class"].reshape(-1, 1)], axis=1)
            elif item == "affine":
                input_record["affine"] = np.eye(3, dtype="float32")
                # print(input_record["affine"].dtype)
        # print(input_record["image_url"], input_record["label"], input_record["image"].shape)
        input_record["im_id"] = np.array(input_record["im_id"], dtype="float32")


class Norm2DImage(Augmenter):
    """
    input: image, ndarray(h, w, rgb)
    output: image, ndarray(h, w, rgb)
    """

    def __init__(self, pNorm):
        self.p = pNorm  # type: NormParam

    def __call__(self, input_record):
        p = self.p

        image = input_record["image"]
        image -= p.mean
        image /= p.std

        input_record["image"] = image


class Resize2DImage(Augmenter):
    """
    input: image, ndarray(h, w, rgb)
    output: image, ndarray(h', w', rgb)
            im_info, ndarray([h', w', scale])
    """

    def __init__(self, pResize):
        self.p = pResize  # type: ResizeParam

    def __call__(self, input_record):
        p = self.p
        resize_h = p.height
        resize_w = p.width

        image = input_record["image"]
        h, w = image.shape[:2]
        scale_h = resize_h / h
        scale_w = resize_w / w

        input_record["image"] = cv2.resize(
            image, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
        input_record["im_info"] = np.array([resize_h, resize_w, scale_h, scale_w], dtype=np.float32)


class ResizeShortWithin2DImage(Augmenter):
    """
    input: image, ndarray(h, w, rgb)
    output: image, ndarray(h', w', rgb)
            im_info, ndarray([h', w', scale])
    """

    def __init__(self, pResize):
        self.p = pResize  # type: ResizeParam

    def __call__(self, input_record):
        p = self.p

        image = input_record["image"]

        short = min(image.shape[:2])
        long = max(image.shape[:2])
        scale = min(p.short / short, p.long / long)
        h, w = image.shape[:2]

        input_record["image"] = cv2.resize(
            image, None, None, scale, scale, interpolation=cv2.INTER_LINEAR)
        input_record["im_info"] = np.array(
            [round(h * scale), round(w * scale), scale], dtype=np.float32)


class RandCrop2DImage(Augmenter):
    """
    input: image, ndarray(h, w, rgb)
    output: image, ndarray(h2, w2, rgb)
    """

    def __init__(self, pCrop):
        self.p = pCrop  # type: CropParam

    def __call__(self, input_record):
        p = self.p

        image = input_record["image"]
        # random select the cropped region
        height, width = image.shape[:2]
        assert 0.0 < p.min_ratio <= p.max_ratio <= 1.0
        min_crop_ratio = p.min_ratio
        max_crop_ratio = p.max_ratio
        h_crop = max(1, int(np.random.uniform(min_crop_ratio, max_crop_ratio) * height))
        w_crop = max(1, int(np.random.uniform(min_crop_ratio, max_crop_ratio) * width))
        h_start = np.random.randint(0, height - h_crop + 1)
        h_end = h_start + h_crop
        w_start = np.random.randint(0, width - w_crop + 1)
        w_end = w_start + w_crop

        # crop image
        im_cropped = image[h_start: h_end, w_start: w_end]
        input_record["image"] = im_cropped


class RandCropHW2DImage(Augmenter):
    """
    input: image, ndarray(h, w, rgb)
    output: image, ndarray(h2, w2, rgb)
    """

    def __init__(self, pCrop):
        self.p = pCrop  # type: CropParam

    def __call__(self, input_record):
        p = self.p

        image = input_record["image"]
        # random select the cropped region
        height, width = image.shape[:2]
        assert p.height <= height and p.width <= width
        h_crop = p.height
        w_crop = p.width
        h_start = np.random.randint(0, height - h_crop + 1)
        h_end = h_start + h_crop
        w_start = np.random.randint(0, width - w_crop + 1)
        w_end = w_start + w_crop

        # crop image
        im_cropped = image[h_start: h_end, w_start: w_end]
        input_record["image"] = im_cropped


class RandErasing2DImage(object):
    """
    input: image, ndarray(h, w, rgb)
    output: image, ndarray(j, w, rgb)
    """

    def __init__(self, pErasing):
        self.p = pErasing  # type: ErasingParam

    def __call__(self, input_record):
        p = self.p
        assert 0.0 <= p.prob <= 1.0
        assert 0.0 < p.min_ratio <= p.max_ratio <= 1.0
        assert 0.0 < p.aspect_ratio < 1.0

        if np.random.uniform() < p.prob:
            image = input_record["image"]

            height, width = image.shape[:2]
            min_erase_ratio = p.min_ratio
            max_erase_ratio = p.max_ratio
            aspect_ratio = p.aspect_ratio
            mean = p.mean

            area = height * width
            erase_area = np.random.uniform(min_erase_ratio, max_erase_ratio) * area
            erase_aspect_ratio = 1.0 / np.random.uniform(aspect_ratio, 1.0 / aspect_ratio)
            w = int(round(np.sqrt(erase_area * erase_aspect_ratio)))
            h = int(round(np.sqrt(erase_area / erase_aspect_ratio)))
            if w < width and h < height:
                x1 = np.random.randint(0, width - w)
                y1 = np.random.randint(0, height - h)
                image[y1: y1 + h, x1: x1 + w, 0] = mean[0]
                image[y1: y1 + h, x1: x1 + w, 1] = mean[1]
                image[y1: y1 + h, x1: x1 + w, 2] = mean[2]
            input_record["image"] = image


class RandBrightness2Dimage(object):
    """
    input: image, ndarray(h, w, rgb)
    output: image, ndarray(j, w, rgb)
    """

    def __init__(self, pBrightness):
        self.p = pBrightness  # type: BrightnessParam

    def __call__(self, input_record):
        p = self.p
        if np.random.uniform() < p.prob:
            image = input_record["image"]
            image_int = image.astype(np.int)
            ratio = 1.0 + np.clip(np.random.randn() * 0.5, a_min=-0.8, a_max=0.8)
            image_int[:, :, 2] = image_int[:, :, 2] * ratio
            image_int = np.clip(image_int, a_min=0, a_max=255)
            input_record["image"] = image_int.astype(np.uint8)


class RandFlip2DImage(Augmenter):
    """
    input: image, ndarray(h, w, rgb)
    output: image, ndarray(h, w, rgb)
    """

    def __init__(self):
        pass

    def __call__(self, input_record):
        if np.random.uniform() > 0.5:
            image = input_record["image"]
            input_record["image"] = image[:, ::-1]
            input_record["flipped"] = True


class Flip2DImage(Augmenter):
    """
    input: image, ndarray(h, w, rgb)
    output: image, ndarray(h, w, rgb)
    """

    def __init__(self):
        pass

    def __call__(self, input_record):
        if "flipped" in input_record and input_record["flipped"]:
            image = input_record["image"]
            input_record["image"] = image[:, ::-1]
        elif "flipped" not in input_record:
            input_record["flipped"] = False


class Pad2DImage(Augmenter):
    """
    input: image, ndarray(h, w, rgb)
    output: image, ndarray(h, w, rgb)
    """

    def __init__(self, pPad):
        self.p = pPad  # type: PadParam

    def __call__(self, input_record):
        p = self.p

        image = input_record["image"]

        h, w = image.shape[:2]
        shape = (p.long, p.short, 3) if input_record["h"] >= input_record["w"] \
            else (p.short, p.long, 3)

        padded_image = np.zeros(shape, dtype=np.float32)
        padded_image[:h, :w] = image

        input_record["image"] = padded_image


class PadHW2DImage(Augmenter):
    """
    input: image, ndarray(h, w, rgb)
    output: image, ndarray(h, w, rgb)
    """

    def __init__(self, pPad):
        self.p = pPad  # type: PadParam

    def __call__(self, input_record):
        p = self.p

        image = input_record["image"]

        h, w = image.shape[:2]
        assert p.height >= h and p.width >= w
        shape = (p.height, p.width, 3)
        h_start = round((p.height - h) / 2)
        w_start = round((p.width - w) / 2)

        padded_image = np.zeros(shape, dtype=np.float32)
        padded_image[h_start: h + h_start, w_start: w + w_start] = image

        input_record["image"] = padded_image


class ConvertImageFromHWCToCHW(Augmenter):
    def __init__(self):
        pass

    def __call__(self, input_record):
        input_record["image"] = input_record["image"].transpose((2, 0, 1))


class RenameRecord(Augmenter):
    def __init__(self, pRename):
        self.p = pRename  # RenameParam

    def __call__(self, input_record):
        mapping = self.p.mapping
        for k, new_k in mapping.items():
            input_record[new_k] = input_record.pop(k)


class RandFlip3DImageBboxJoint(Augmenter):
    """
    input: image, ndarray(h, w, rgb)
           gt_bbox, ndarray(4,)
           gt_joints_3d, ndarray(num_joints, 3)
           gt_joints_vis, ndarray(num_joints,)
    output: image, ndarray(h, w, rgb)
            gt_bbox, ndarray(4,)
            gt_joints_3d, ndarray(num_joints, 3)
            gt_joints_vis, ndarray(num_joints,)
            flipped, bool
    """

    def __init__(self, pFlip):
        self.p = pFlip  # FlipParam

    def __call__(self, input_record):
        joint_pairs = self.p.joint_pairs

        assert not input_record["flipped"]
        if np.random.uniform() > 0.5:
            image = input_record["image"]
            gt_bbox = input_record["gt_bbox"]
            joints_3d = input_record["gt_joints_3d"]
            joints_vis = input_record["gt_joints_vis"]

            input_record["image"] = image[:, ::-1]

            flipped_bbox = gt_bbox.copy()
            h, w = image.shape[:2]
            flipped_bbox[0] = (w - 1) - gt_bbox[2]
            flipped_bbox[2] = (w - 1) - gt_bbox[0]
            input_record["gt_bbox"] = flipped_bbox

            flipped_joints = joints_3d.copy()
            flipped_joints_vis = joints_vis.copy()
            flipped_joints[:, 0] = (w - 1) - flipped_joints[:, 0]
            for pair in joint_pairs:
                flipped_joints[pair[0], :], flipped_joints[pair[1], :] = \
                    flipped_joints[pair[1], :].copy(), flipped_joints[pair[0], :].copy()
                flipped_joints_vis[pair[0]], flipped_joints_vis[pair[1]] = \
                    flipped_joints_vis[pair[1]].copy(), flipped_joints_vis[pair[0]].copy()
            input_record["gt_joints_3d"] = flipped_joints
            input_record["gt_joints_vis"] = flipped_joints_vis

            input_record["flipped"] = True


class Center3DMatrix(Augmenter):
    """
    input: gt_bbox, ndarray(4,)
           affine, ndarray(3, 3)
    output: affine, ndarray(3, 3)
    """

    def __init__(self):
        pass

    def __call__(self, input_record):
        affine = input_record["affine"]

        x1, y1, x2, y2 = input_record["gt_bbox"]
        center_x = (x1 + x2) * 0.5
        center_y = (y1 + y2) * 0.5
        center = np.array([[1, 0, -center_x], [0, 1, -center_y], [0, 0, 1]], dtype="float32")
        affine = center @ affine

        input_record["affine"] = affine


class Rotate3DMatrix(Augmenter):
    """
    input: affine, ndarray(3, 3)
    output: affine, ndarray(3, 3)
    """

    def __init__(self, pRotate):
        self.p = pRotate  # RotateParam

    def __call__(self, input_record):
        rp = self.p.rotation_p
        rf = self.p.rotation_factor
        center_y = self.p.height * 0.5
        center_x = self.p.width * 0.5
        angle = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) \
            if np.random.uniform() <= rp else 0.0

        affine = input_record["affine"]

        theta = angle / 360 * 2 * np.pi
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        rotate = np.array([
            [cos_theta, -sin_theta, center_x * (1 - cos_theta) + center_y * sin_theta],
            [sin_theta, cos_theta, -center_x * sin_theta + center_y * (1 - cos_theta)],
            [0, 0, 1]],
            dtype="float32"
        )
        affine = rotate @ affine

        input_record["affine"] = affine
        input_record["angle"] = np.ones((1,)) * angle


class Rotate3DMatrixV2(Augmenter):
    """
    input: affine, ndarray(3, 3)
    output: affine, ndarray(3, 3)
    """

    def __init__(self, pRotate):
        self.p = pRotate  # RotateParam

    def __call__(self, input_record):
        rp = self.p.rotation_p
        rf = self.p.rotation_factor
        x1, y1, x2, y2 = input_record["gt_bbox"]
        center_x = (x1 + x2) * 0.5
        center_y = (y1 + y2) * 0.5
        angle = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) \
            if np.random.uniform() <= rp else 0.0

        affine = input_record["affine"]

        theta = angle / 360 * 2 * np.pi
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        rotate = np.array([
            [cos_theta, -sin_theta, center_x * (1 - cos_theta) + center_y * sin_theta],
            [sin_theta, cos_theta, -center_x * sin_theta + center_y * (1 - cos_theta)],
            [0, 0, 1]],
            dtype="float32"
        )
        affine = rotate @ affine

        input_record["affine"] = affine
        input_record["angle"] = np.ones((1,)) * angle


class Rescale3DMatrix(Augmenter):
    """
    input: affine, ndarray(3, 3)
    output: affine, ndarray(3, 3)
    """

    def __init__(self, pRescale):
        self.p = pRescale  # RescaleParam

    def __call__(self, input_record):
        sf = self.p.scaling_factor
        center_y = self.p.height * 0.5
        center_x = self.p.width * 0.5
        scale = np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)

        affine = input_record["affine"]

        rescale = np.array([
            [scale, 0, (1 - scale) * center_x],
            [0, scale, (1 - scale) * center_y],
            [0, 0, 1]],
            dtype="float32"
        )
        affine = rescale @ affine

        input_record["affine"] = affine


class Rescale3DMatrixV2(Augmenter):
    """
    input: affine, ndarray(3, 3)
    output: affine, ndarray(3, 3)
    """

    def __init__(self, pRescale):
        self.p = pRescale  # RescaleParam

    def __call__(self, input_record):
        sf = self.p.scaling_factor
        center_y = self.p.height * 0.5
        center_x = self.p.width * 0.5
        scale = 1.0 / np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)

        affine = input_record["affine"]

        rescale = np.array([
            [scale, 0, (1 - scale) * center_x],
            [0, scale, (1 - scale) * center_y],
            [0, 0, 1]],
            dtype="float32"
        )
        affine = rescale @ affine

        input_record["affine"] = affine


class Rescale3DMatrixV3(Augmenter):
    """
    input: affine, ndarray(3, 3)
    output: affine, ndarray(3, 3)
    """

    def __init__(self, pRescale):
        self.p = pRescale  # RescaleParam

    def __call__(self, input_record):
        sf = self.p.scaling_factor
        x1, y1, x2, y2 = input_record["gt_bbox"]
        center_x = (x1 + x2) * 0.5
        center_y = (y1 + y2) * 0.5
        scale = np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)

        affine = input_record["affine"]

        rescale = np.array([
            [scale, 0, (1 - scale) * center_x],
            [0, scale, (1 - scale) * center_y],
            [0, 0, 1]],
            dtype="float32"
        )
        affine = rescale @ affine

        input_record["affine"] = affine


class Crop3DMatrix(Augmenter):
    """
    input: gt_bbox, ndarray(4,)
           affine, ndarray(3, 3)
    output: affine, ndarray(3, 3)
    """

    def __init__(self, pCrop):
        self.p = pCrop

    def __call__(self, input_record):
        bbox_expand = self.p.bbox_expand
        resize_h = self.p.height
        resize_w = self.p.width

        x1, y1, x2, y2 = input_record["gt_bbox"]
        affine = input_record["affine"]

        center_x = (x1 + x2) * 0.5
        center_y = (y1 + y2) * 0.5
        bbox_w = (x2 - x1) * bbox_expand
        bbox_h = (y2 - y1) * bbox_expand

        ratio = resize_w / resize_h  # float
        if bbox_h * ratio > bbox_w:
            bbox_w = bbox_h * ratio
        elif bbox_w / ratio > bbox_h:
            bbox_h = bbox_w / ratio

        scale_w = resize_w / bbox_w  # float
        scale_h = resize_h / bbox_h  # float

        crop = np.array([
            [scale_w, 0, -center_x * scale_w + resize_w * 0.5],
            [0, scale_h, -center_y * scale_h + resize_h * 0.5],
            [0, 0, 1]],
            dtype="float32"
        )

        affine = crop @ affine

        input_record["affine"] = affine


class Crop3DMatrixV2(Augmenter):
    """
    input: gt_bbox, ndarray(4,)
           affine, ndarray(3, 3)
    output: affine, ndarray(3, 3)
    """

    def __init__(self, pCrop):
        self.p = pCrop

    def __call__(self, input_record):
        bbox_expand = self.p.bbox_expand
        resize_h = self.p.height
        resize_w = self.p.width

        x1, y1, x2, y2 = input_record["gt_bbox"]
        affine = input_record["affine"]

        center_x = (x1 + x2) * 0.5
        center_y = (y1 + y2) * 0.5
        bbox_w = (x2 - x1) * bbox_expand
        bbox_h = (y2 - y1) * bbox_expand

        # ratio = resize_w / resize_h  # float
        # if bbox_h * ratio > bbox_w:
        #     bbox_w = bbox_h * ratio
        # elif bbox_w / ratio > bbox_h:
        #     bbox_h = bbox_w / ratio

        scale_w = resize_w / bbox_w  # float
        scale_h = resize_h / bbox_h  # float

        crop = np.array([
            [scale_w, 0, -center_x * scale_w + resize_w * 0.5],
            [0, scale_h, -center_y * scale_h + resize_h * 0.5],
            [0, 0, 1]],
            dtype="float32"
        )

        affine = crop @ affine

        input_record["affine"] = affine


class Crop3DMatrixV3(Augmenter):
    """
    input: gt_bbox, ndarray(4,)
           affine, ndarray(3, 3)
    output: affine, ndarray(3, 3)
    """

    def __init__(self, pCrop):
        self.p = pCrop

    def __call__(self, input_record):
        bbox_expand = self.p.bbox_expand
        resize_h = self.p.height
        resize_w = self.p.width

        x1, y1, x2, y2 = input_record["gt_bbox"].copy()
        affine = input_record["affine"]
        corners = [[x1, y1], [x1, y2], [x2, y1], [x2, y2]]
        new_pts = []
        for pt in corners:
            new_pt = np.array([pt[0], pt[1], 1.]).T
            new_pt = np.dot(affine[:2], new_pt)
            new_pts.append(new_pt[:2])
        new_pts = np.array(new_pts)
        x1 = np.min(new_pts[:, 0])
        x2 = np.max(new_pts[:, 0])
        y1 = np.min(new_pts[:, 1])
        y2 = np.max(new_pts[:, 1])

        center_x = (x1 + x2) * 0.5
        center_y = (y1 + y2) * 0.5
        bbox_w = (x2 - x1) * bbox_expand
        bbox_h = (y2 - y1) * bbox_expand

        # ratio = resize_w / resize_h  # float
        # if bbox_h * ratio > bbox_w:
        #     bbox_w = bbox_h * ratio
        # elif bbox_w / ratio > bbox_h:
        #     bbox_h = bbox_w / ratio

        scale_w = resize_w / bbox_w  # float
        scale_h = resize_h / bbox_h  # float

        crop = np.array([
            [scale_w, 0, -center_x * scale_w + resize_w * 0.5],
            [0, scale_h, -center_y * scale_h + resize_h * 0.5],
            [0, 0, 1]],
            dtype="float32"
        )

        affine = crop @ affine

        input_record["affine"] = affine


class Affine3DImageJoint(Augmenter):
    """
    input: image, ndarray(h, w, rgb)
           gt_bbox, ndarray(4,)
           gt_joints_3d, ndarray(num_joints, 3)
           gt_joints_vis, ndarray(num_joints,)
           affine, ndarray(3, 3)
    output: image, ndarray(h', w', rgb)
            gt_bbox, ndarray(4,)
            gt_joints_3d, ndarray(num_joints, 3)
    """

    def __init__(self, pAffine):
        self.p = pAffine

    def __call__(self, input_record):
        num_joints = self.p.num_joints
        resize_h = self.p.height
        resize_w = self.p.width

        image = input_record["image"]
        # gt_bbox = input_record["gt_bbox"].copy()
        joints_3d = input_record["gt_joints_3d"].copy()
        joints_vis = input_record["gt_joints_vis"]
        affine = input_record["affine"]

        image = cv2.warpAffine(image, affine[:2], (resize_w, resize_h), flags=cv2.INTER_LINEAR)
        input_record["image"] = image

        # for i in range(2):
        #     new_pt = np.array([gt_bbox[i * 2], gt_bbox[i * 2 + 1], 1.]).T
        #     new_pt = np.dot(affine[:2], new_pt)
        #     gt_bbox[i * 2: (i + 1) * 2] = new_pt[:2]
        # input_record["gt_bbox"] = gt_bbox

        for i in range(num_joints):
            if joints_vis[i] > 0.0:
                new_pt = np.array([joints_3d[i, 0], joints_3d[i, 1], 1.]).T
                new_pt = np.dot(affine[:2], new_pt)
                joints_3d[i][:2] = new_pt[:2]
        input_record["gt_joints_3d"] = joints_3d


class Affine3DImage(Augmenter):
    """
    input: image, ndarray(h, w, rgb)
           gt_bbox, ndarray(4,)
           affine, ndarray(3, 3)
    output: image, ndarray(h', w', rgb)
            gt_bbox, ndarray(4,)
    """

    def __init__(self, pAffine):
        self.p = pAffine

    def __call__(self, input_record):
        resize_h = self.p.height
        resize_w = self.p.width

        image = input_record["image"]
        # gt_bbox = input_record["gt_bbox"].copy()
        affine = input_record["affine"]

        image = cv2.warpAffine(image, affine[:2], (resize_w, resize_h), flags=cv2.INTER_LINEAR)
        input_record["image"] = image

        # for i in range(2):
        #     new_pt = np.array([gt_bbox[i * 2], gt_bbox[i * 2 + 1], 1.]).T
        #     new_pt = np.dot(affine[:2], new_pt)
        #     gt_bbox[i * 2: (i + 1) * 2] = new_pt[:2]
        # input_record["gt_bbox"] = gt_bbox


class GenGaussianTarget(Augmenter):
    """
    input: gt_joints_3d, ndarray(num_joints, 3)
           gt_joints_vis, ndarray(num_joints,)
    output: target, ndarray(num_joints, h, w)
            target_weight, ndarray(num_joints, 1, 1)
    """

    def __init__(self, pTarget):
        self.p = pTarget

    def __call__(self, input_record):
        num_joints = self.p.num_joints
        height = self.p.height
        width = self.p.width
        heatmap_height = self.p.heatmap_height
        heatmap_width = self.p.heatmap_width
        sigma = self.p.sigma

        joints_3d = input_record["gt_joints_3d"].copy()
        joints_vis = input_record["gt_joints_vis"].copy()

        target = np.zeros((num_joints, heatmap_height, heatmap_width), dtype=np.float32)
        target_weight = np.array(joints_vis, dtype=np.float32)
        h_size = sigma * 3
        height_stride = height / heatmap_height
        width_stride = width / heatmap_width

        for i in range(num_joints):
            mu_x = int(joints_3d[i, 0] / width_stride + 0.5)
            mu_y = int(joints_3d[i, 1] / height_stride + 0.5)
            # check that any part of the gaussian is in-bounds
            ul = [int(mu_x - h_size), int(mu_y - h_size)]
            br = [int(mu_x + h_size + 1), int(mu_y + h_size + 1)]
            if ul[0] >= heatmap_width or ul[1] >= heatmap_height or br[0] < 0 or br[1] < 0:
                # if not, just return the image as is
                target_weight[i] = 0
                continue

            size = 2 * h_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            # the gaussian is not normalized, we want the center value to equal 1
            g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

            # usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], heatmap_width) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], heatmap_height) - ul[1]
            # image range
            img_x = max(0, ul[0]), min(br[0], heatmap_width)
            img_y = max(0, ul[1]), min(br[1], heatmap_height)

            v = target_weight[i]
            if v > 0.5:
                target[i][img_y[0]: img_y[1], img_x[0]: img_x[1]] = \
                    g[g_y[0]: g_y[1], g_x[0]: g_x[1]]

        input_record["target"] = target
        input_record["target_weight"] = target_weight.reshape((num_joints, 1, 1))


class GenDensePoseTarget(Augmenter):
    """
    input: image, ndarray(h, w, rgb)
           affine, ndarray(3, 3)
    output: mask, ndarray(h', w', cls)
    """

    def __init__(self, pDPTarget):
        self.p = pDPTarget

    @staticmethod
    def _gen_mask(polys):
        import pycocotools.mask as mask_util
        num_masks = len(polys)
        mask = np.zeros((256, 256, num_masks), dtype="uint8")
        for i in range(num_masks):
            if polys[i]:
                current_mask = mask_util.decode(polys[i])
                mask[current_mask > 0, i] = 1
        return mask

    def __call__(self, input_record):
        resize_h = self.p.height
        resize_w = self.p.width
        resize_hh = self.p.heatmap_height
        resize_hw = self.p.heatmap_width

        image = input_record["image"]
        x1, y1, x2, y2 = np.round(input_record["gt_bbox"]).astype("int")
        # print(input_record["gt_bbox"], x1, y1, x2, y2)
        affine = input_record["affine"]
        dp_masks = input_record["dp_masks"]
        dp_x = input_record["dp_x"]
        dp_y = input_record["dp_y"]
        dp_I = input_record["dp_I"]
        dp_U = input_record["dp_U"]
        dp_V = input_record["dp_V"]

        # decode mask
        mask_image_shape = (image.shape[0], image.shape[1], len(dp_masks))
        mask_image = np.zeros(mask_image_shape, dtype="uint8")
        # print("mask_image", mask_image.shape)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(x2, image.shape[1])
        y2 = min(y2, image.shape[0])
        mask_target = GenDensePoseTarget._gen_mask(dp_masks)
        # print(x1, y1, x2, y2)
        # print(int(x2 - x1), int(y2 - y1))
        mask_target = cv2.resize(mask_target, (int(x2 - x1), int(y2 - y1)))
        mask_image[y1: y2, x1: x2] = mask_target
        mask = cv2.warpAffine(mask_image, affine[:2], (resize_w, resize_h))
        mask = cv2.resize(mask, (resize_hw, resize_hh))
        argmax_mask = np.argmax(mask, axis=-1)
        max_mask = np.max(mask, axis=-1)
        index_mask = np.where(max_mask > 0, argmax_mask, -1) + 1
        input_record["dp_masks"] = index_mask.astype("float32")

        # decode dp_I, dp_U, dp_V, dp_x, dp_y
        dp_I_ = -np.ones((192,), dtype="float32")
        dp_U_ = np.zeros((192,), dtype="float32")
        dp_V_ = np.zeros((192,), dtype="float32")
        dp_x_ = np.zeros((192,), dtype="float32")
        dp_y_ = np.zeros((192,), dtype="float32")
        pts = np.ones((len(dp_x), 3))
        pts[:, 0] = [x / 255.0 * (x2 - x1) + x1 for x in dp_x]
        pts[:, 1] = [y / 255.0 * (y2 - y1) + y1 for y in dp_y]
        pts = np.dot(affine, pts.T).T
        i = 0
        for j, pt in enumerate(pts):
            if 0 <= pt[0] <= resize_w - 1 and 0 <= pt[1] <= resize_h - 1:
                dp_I_[i] = dp_I[j]
                dp_U_[i] = dp_U[j]
                dp_V_[i] = dp_V[j]
                dp_x_[i] = pt[0] / (resize_w - 1.0) * 2.0 - 1.0
                dp_y_[i] = pt[1] / (resize_h - 1.0) * 2.0 - 1.0
                i += 1

        input_record["dp_I"] = dp_I_
        input_record["dp_U"] = dp_U_
        input_record["dp_V"] = dp_V_
        input_record["dp_x"] = dp_x_
        input_record["dp_y"] = dp_y_


class RandFlip3DImageBboxDensePose(Augmenter):
    """
    input: image, ndarray(h, w, rgb)
           gt_bbox, ndarray(4,)
           dp_x, ndarray(n,)
           dp_masks, list or dict
    output: image, ndarray(h, w, rgb)
            gt_bbox, ndarray(4,)
            dp_x, ndarray(n,)
            dp_masks, list or dict
            flipped, bool
    """

    def __init__(self, pFlip):
        from scipy.io import loadmat
        self.p = pFlip  # FlipParam
        # self.SemanticMaskSymmetries = [0, 1, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 14]
        self.mask_symmetry = pFlip.mask_symmetry
        self.index_symmetry = pFlip.index_symmetry
        self.uv_symmetry = loadmat(pFlip.uv_symmetry_file)

    @staticmethod
    def _flip_poly(poly, width):
        flipped_poly = np.array(poly)
        flipped_poly[0::2] = width - np.array(poly[0::2]) - 1
        return flipped_poly.tolist()

    @staticmethod
    def _flip_rle(rle, height, width):
        import pycocotools.mask as mask_util
        # print(rle)
        if "counts" in rle and type(rle['counts']) == list:
            # Magic RLE format handling painfully discovered by looking at the
            # COCO API showAnns function.
            rle = mask_util.frPyObjects([rle], height, width)
            # print(rle)
        mask = mask_util.decode(rle)
        # print(mask)
        mask = mask[:, ::-1]
        rle = mask_util.encode(np.array(mask, order='F', dtype=np.uint8))
        return rle

    def __call__(self, input_record):
        assert not input_record["flipped"]
        if np.random.uniform() > 0.5:
            image = input_record["image"]
            gt_bbox = input_record["gt_bbox"]
            dp_I = input_record["dp_I"]
            dp_U = input_record["dp_U"]
            dp_V = input_record["dp_V"]
            dp_x = input_record["dp_x"]
            dp_masks = input_record["dp_masks"]

            # image
            input_record["image"] = image[:, ::-1]
            # gt_bbox
            flipped_bbox = gt_bbox.copy()
            h, w = image.shape[:2]
            flipped_bbox[0] = (w - 1) - gt_bbox[2]
            flipped_bbox[2] = (w - 1) - gt_bbox[0]
            input_record["gt_bbox"] = flipped_bbox
            # dp_I
            flipped_dp_I = []
            flipped_dp_U = []
            flipped_dp_V = []
            for index, I in enumerate(dp_I):
                assert (I - 1 - int(I - 1)) == 0
                flipped_dp_I.append(self.index_symmetry[int(I - 1)])
                U_loc = int(dp_U[index] * 255)
                V_loc = int(dp_V[index] * 255)
                flipped_dp_U.append(
                    self.uv_symmetry['U_transforms'][0, int(I - 1)][V_loc, U_loc])
                flipped_dp_V.append(
                    self.uv_symmetry['V_transforms'][0, int(I - 1)][V_loc, U_loc])
            input_record["dp_I"] = flipped_dp_I
            input_record["dp_U"] = flipped_dp_U
            input_record["dp_V"] = flipped_dp_V
            # dp_x
            flipped_dp_x = [255.0 - x for x in dp_x]
            input_record["dp_x"] = flipped_dp_x
            # dp_masks
            assert len(dp_masks) == 14
            flipped_dp_masks = []
            for i in self.mask_symmetry:
                mask = dp_masks[i]
                if type(mask) == list:
                    # Polygon
                    f_masks = [RandFlip3DImageBboxDensePose._flip_poly(poly, w) for poly in mask]
                    flipped_dp_masks.append(f_masks)
                else:
                    # RLE
                    flipped_dp_masks.append(RandFlip3DImageBboxDensePose._flip_rle(mask, h, w))
            input_record["dp_masks"] = flipped_dp_masks

            input_record["flipped"] = True
