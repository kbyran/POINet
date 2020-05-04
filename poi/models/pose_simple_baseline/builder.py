import mxnet as mx


class Builder(object):
    def __init__(self):
        pass

    @staticmethod
    def get_train_symbol(backbone, neck, head):
        data = mx.sym.var("data")
        target = mx.sym.var("target")
        target_weight = mx.sym.var("target_weight")
        feat = backbone.get_feature(data)
        heatmap = neck.get_heatmap(feat)
        loss = head.get_loss(heatmap, target, target_weight)
        return mx.sym.Group([loss, mx.sym.BlockGrad(heatmap, name="heatmap_blockgrad")])
        # max_idx, max_val = head.get_output(heatmap)
        # return mx.sym.Group([loss, heatmap, max_idx, max_val])

    @staticmethod
    def get_test_symbol(backbone, neck, head):
        data = mx.sym.var("data")
        im_id = mx.sym.var("im_id")
        rec_id = mx.sym.var("rec_id")
        affine = mx.sym.var("affine")

        feat = backbone.get_feature(data)
        heatmap = neck.get_heatmap(feat)
        max_idx, max_val = head.get_output(heatmap)

        return mx.sym.Group([rec_id, im_id, affine, max_idx, max_val])

    @staticmethod
    def get_export_symbol(backbone, neck, head):
        data = mx.sym.var("data")

        feat = backbone.get_feature(data)
        heatmap = neck.get_heatmap(feat)
        max_idx, max_val = head.get_output(heatmap)

        return mx.sym.Group([max_idx, max_val])


class Neck(object):
    def __init__(self, pNeck):
        self.p = pNeck
        self.heatmap = None

    def get_heatmap(self, feature):
        num_joints = self.p.num_joints
        num_deconv = self.p.num_deconv
        num_deconv_filter = self.p.num_deconv_filter
        num_deconv_kernel = self.p.num_deconv_kernel
        conv_kernel = self.p.conv_kernel

        for i in range(num_deconv):
            deconv_weight = mx.sym.var(
                "deconv{}_weight".format(i), init=mx.init.Normal(sigma=0.001)
            )
            feature = mx.sym.Deconvolution(
                data=feature,
                weight=deconv_weight,
                kernel=(num_deconv_kernel[i], num_deconv_kernel[i]),
                stride=(2, 2),
                pad=(1, 1),
                adj=(0, 0),
                num_filter=num_deconv_filter[i],
                no_bias=True,
                name="deconv{}".format(i)
            )
            bn = mx.sym.BatchNorm(
                data=feature,
                fix_gamma=False,
                use_global_stats=False,
                eps=1e-5,
                cudnn_off=True,
                name="deconv_bn{}".format(i)
            )
            relu = mx.sym.relu(data=bn, name="deconv_relu{}".format(i))
            feature = relu

        heatmap_weight = mx.sym.var(
            "final_conv_weight", init=mx.init.Normal(sigma=0.001)
        )
        heatmap = mx.sym.Convolution(
            data=feature,
            weight=heatmap_weight,
            kernel=(conv_kernel, conv_kernel),
            stride=(1, 1),
            num_filter=num_joints,
            name="final_conv"
        )
        self.heatmap = heatmap

        return self.heatmap


class Head(object):
    def __init__(self, pHead):
        self.p = pHead

    def get_loss(self, heatmap, target, target_weight):
        square_loss = mx.sym.square(heatmap - target, name="square_loss")
        weighted_loss = mx.sym.broadcast_mul(square_loss, target_weight) * 0.5
        mean_loss = mx.sym.mean(weighted_loss, axis=(2, 3))
        sum_loss = mx.sym.sum(mean_loss, axis=1)
        norm_loss = sum_loss / (mx.sym.sum(target_weight, axis=(1, 2, 3)) + 1e-6)
        loss = mx.sym.MakeLoss(norm_loss, name="l2_loss", normalization="batch")
        return loss

    def get_output(self, heatmap):
        batch_size = self.p.batch_image
        num_joints = self.p.num_joints
        heatmap_height = self.p.heatmap_height
        heatmap_width = self.p.heatmap_width
        from poi.ops.fuse.index_ops import heatmap2points
        max_idx, max_val = heatmap2points(
            mx.sym, heatmap, batch_size, num_joints, heatmap_height, heatmap_width)

        return mx.sym.identity(max_idx, name="max_idx"), mx.sym.identity(max_val, name="max_val")
