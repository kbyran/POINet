import mxnet as mx


class Builder(object):
    def __init__(self):
        pass

    @staticmethod
    def get_train_symbol(backbone, neck, head):
        data = mx.sym.var("data")
        dp_masks = mx.sym.var("dp_masks")
        dp_I = mx.sym.var("dp_I")
        dp_U = mx.sym.var("dp_U")
        dp_V = mx.sym.var("dp_V")
        dp_x = mx.sym.var("dp_x")
        dp_y = mx.sym.var("dp_y")
        feat = backbone.get_feature(data)
        feat = neck.get_feature(feat)
        ann_index_lowres = neck.get_ann_index_lowres(feat)
        index_uv_lowres = neck.get_index_uv_lowres(feat)
        u_lowres = neck.get_u_lowres(feat)
        v_lowres = neck.get_v_lowres(feat)
        seg_loss, index_uv_loss, u_loss, v_loss = head.get_loss(
            ann_index_lowres, index_uv_lowres, u_lowres, v_lowres,
            dp_masks, dp_I, dp_U, dp_V, dp_x, dp_y
        )
        return mx.sym.Group([seg_loss, index_uv_loss, u_loss, v_loss])

    @staticmethod
    def get_test_symbol(backbone, neck, head):
        data = mx.sym.var("data")
        im_id = mx.sym.var("im_id")
        rec_id = mx.sym.var("rec_id")
        affine = mx.sym.var("affine")

        feat = backbone.get_feature(data)
        feat = neck.get_feature(feat)
        ann_index_lowres = neck.get_ann_index_lowres(feat)
        index_uv_lowres = neck.get_index_uv_lowres(feat)
        u_lowres = neck.get_u_lowres(feat)
        v_lowres = neck.get_v_lowres(feat)
        ann_index_lowres, index_uv_lowres, u_lowres, v_lowres = head.get_output(
            ann_index_lowres, index_uv_lowres, u_lowres, v_lowres
        )
        outputs = mx.sym.Group(
            [rec_id, im_id, affine, ann_index_lowres, index_uv_lowres, u_lowres, v_lowres]
        )
        return outputs

    @staticmethod
    def get_export_symbol(backbone, neck, head):
        data = mx.sym.var("data")

        feat = backbone.get_feature(data)
        feat = neck.get_feature(feat)
        ann_index_lowres = neck.get_ann_index_lowres(feat)
        index_uv_lowres = neck.get_index_uv_lowres(feat)
        u_lowres = neck.get_u_lowres(feat)
        v_lowres = neck.get_v_lowres(feat)
        ann_index_lowres, index_uv_lowres, u_lowres, v_lowres = head.get_output(
            ann_index_lowres, index_uv_lowres, u_lowres, v_lowres
        )
        outputs = mx.sym.Group(
            [ann_index_lowres, index_uv_lowres, u_lowres, v_lowres]
        )
        return outputs


class Neck(object):
    def __init__(self, pNeck):
        self.p = pNeck
        self.heatmap = None

    def get_feature(self, feature):
        num_deconv = self.p.num_deconv
        num_deconv_filter = self.p.num_deconv_filter
        num_deconv_kernel = self.p.num_deconv_kernel

        for i in range(num_deconv):
            feature = mx.sym.Deconvolution(
                data=feature,
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

        self.feature = feature
        return self.feature

    def get_ann_index_lowres(self, feature):
        num_masks = self.p.num_masks
        num_conv = self.p.num_conv
        num_conv_filter = self.p.num_conv_filter
        num_conv_kernel = self.p.num_conv_kernel

        for i in range(num_conv):
            feature = mx.sym.Convolution(
                data=feature,
                kernel=(num_conv_kernel[i], num_conv_kernel[i]),
                stride=(1, 1),
                pad=(1, 1),
                num_filter=num_conv_filter[i],
                no_bias=True,
                name="ann_index_conv{}".format(i)
            )
            bn = mx.sym.BatchNorm(
                data=feature,
                fix_gamma=False,
                use_global_stats=False,
                eps=1e-5,
                cudnn_off=True,
                name="ann_index_conv_bn{}".format(i)
            )
            relu = mx.sym.relu(data=bn, name="ann_index_conv_relu{}".format(i))
            feature = relu

        ann_index_lowres = mx.sym.Convolution(
            data=feature,
            kernel=(1, 1),
            stride=(1, 1),
            num_filter=num_masks + 1,
            name="ann_index_conv"
        )
        self.ann_index_lowres = ann_index_lowres
        return self.ann_index_lowres

    def get_index_uv_lowres(self, feature):
        num_patches = self.p.num_patches
        num_conv = self.p.num_conv
        num_conv_filter = self.p.num_conv_filter
        num_conv_kernel = self.p.num_conv_kernel

        for i in range(num_conv):
            feature = mx.sym.Convolution(
                data=feature,
                kernel=(num_conv_kernel[i], num_conv_kernel[i]),
                stride=(1, 1),
                pad=(1, 1),
                num_filter=num_conv_filter[i],
                no_bias=True,
                name="index_uv_conv{}".format(i)
            )
            bn = mx.sym.BatchNorm(
                data=feature,
                fix_gamma=False,
                use_global_stats=False,
                eps=1e-5,
                cudnn_off=True,
                name="index_uv_conv_bn{}".format(i)
            )
            relu = mx.sym.relu(data=bn, name="index_uv_conv_relu{}".format(i))
            feature = relu

        index_uv_lowres = mx.sym.Convolution(
            data=feature,
            kernel=(1, 1),
            stride=(1, 1),
            num_filter=num_patches + 1,
            name="index_uv_conv"
        )
        self.index_uv_lowres = index_uv_lowres
        return self.index_uv_lowres

    def get_u_lowres(self, feature):
        num_patches = self.p.num_patches
        num_conv = self.p.num_conv
        num_conv_filter = self.p.num_conv_filter
        num_conv_kernel = self.p.num_conv_kernel

        for i in range(num_conv):
            feature = mx.sym.Convolution(
                data=feature,
                kernel=(num_conv_kernel[i], num_conv_kernel[i]),
                stride=(1, 1),
                pad=(1, 1),
                num_filter=num_conv_filter[i],
                no_bias=True,
                name="u_conv{}".format(i)
            )
            bn = mx.sym.BatchNorm(
                data=feature,
                fix_gamma=False,
                use_global_stats=False,
                eps=1e-5,
                cudnn_off=True,
                name="u_conv_bn{}".format(i)
            )
            relu = mx.sym.relu(data=bn, name="u_conv_relu{}".format(i))
            feature = relu

        u_lowres = mx.sym.Convolution(
            data=feature,
            kernel=(1, 1),
            stride=(1, 1),
            num_filter=num_patches + 1,
            name="u_conv"
        )
        self.u_lowres = u_lowres
        return self.u_lowres

    def get_v_lowres(self, feature):
        num_patches = self.p.num_patches
        num_conv = self.p.num_conv
        num_conv_filter = self.p.num_conv_filter
        num_conv_kernel = self.p.num_conv_kernel

        for i in range(num_conv):
            feature = mx.sym.Convolution(
                data=feature,
                kernel=(num_conv_kernel[i], num_conv_kernel[i]),
                stride=(1, 1),
                pad=(1, 1),
                num_filter=num_conv_filter[i],
                no_bias=True,
                name="v_conv{}".format(i)
            )
            bn = mx.sym.BatchNorm(
                data=feature,
                fix_gamma=False,
                use_global_stats=False,
                eps=1e-5,
                cudnn_off=True,
                name="v_bn{}".format(i)
            )
            relu = mx.sym.relu(data=bn, name="v_relu{}".format(i))
            feature = relu

        v_lowres = mx.sym.Convolution(
            data=feature,
            kernel=(1, 1),
            stride=(1, 1),
            num_filter=num_patches + 1,
            name="v_conv"
        )
        self.v_lowres = v_lowres
        return self.v_lowres


class Head(object):
    def __init__(self, pHead):
        self.p = pHead

    def get_loss(self, ann_index_lowres, index_uv_lowres, u_lowres, v_lowres,
                 dp_masks, dp_I, dp_U, dp_V, dp_x, dp_y):
        seg_grad_scale = self.p.seg_grad_scale
        index_grad_scale = self.p.index_grad_scale
        u_grad_scale = self.p.u_grad_scale
        v_grad_scale = self.p.v_grad_scale

        # grid
        dp_x_reshape = mx.sym.reshape(dp_x, shape=(0, 1, -1, 1))
        dp_y_reshape = mx.sym.reshape(dp_y, shape=(0, 1, -1, 1))
        dp_grid = mx.sym.concat(dp_x_reshape, dp_y_reshape, dim=1)

        # sample
        index_uv_interp = mx.sym.BilinearSampler(data=index_uv_lowres, grid=dp_grid)
        index_uv_interp_reshape = mx.sym.reshape(index_uv_interp, shape=(0, 0, -1))
        u_interp = mx.sym.BilinearSampler(data=u_lowres, grid=dp_grid)
        u_interp_reshape = mx.sym.reshape(u_interp, shape=(0, 0, -1))
        u_interp_transpose = mx.sym.transpose(u_interp_reshape, (0, 2, 1))
        u_interp_pick = mx.sym.pick(u_interp_transpose, dp_I)
        v_interp = mx.sym.BilinearSampler(data=v_lowres, grid=dp_grid)
        v_interp_reshape = mx.sym.reshape(v_interp, shape=(0, 0, -1))
        v_interp_transpose = mx.sym.transpose(v_interp_reshape, (0, 2, 1))
        v_interp_pick = mx.sym.pick(v_interp_transpose, dp_I)

        # seg loss
        # ann_index_lowres, dp_masks
        seg_loss = mx.sym.SoftmaxOutput(
            data=ann_index_lowres,
            label=dp_masks,
            grad_scale=seg_grad_scale,
            multi_output=True,
            normalization="batch",
            name="seg_loss"
        )

        # index uv loss
        index_uv_loss = mx.sym.SoftmaxOutput(
            data=index_uv_interp_reshape,
            label=dp_I,
            grad_scale=index_grad_scale,
            ignore_label=-1,
            multi_output=True,
            use_ignore=True,
            normalization="valid",
            name="index_uv_loss"
        )

        # reg loss weight
        reg_weight = dp_I > -1
        reg_weight = mx.sym.broadcast_div(reg_weight, (mx.sym.sum(reg_weight) + 1e-14))
        # u loss
        u_l1 = reg_weight * mx.sym.smooth_l1(u_interp_pick - dp_U, name="u_l1")
        u_loss = mx.sym.MakeLoss(u_l1, grad_scale=u_grad_scale, name="u_loss")
        # v loss
        v_l1 = reg_weight * mx.sym.smooth_l1(v_interp_pick - dp_V, name="v_l1")
        v_loss = mx.sym.MakeLoss(v_l1, grad_scale=v_grad_scale, name="v_loss")
        return seg_loss, index_uv_loss, u_loss, v_loss

    def get_output(self, ann_index_lowres, index_uv_lowres, u_lowres, v_lowres):
        return mx.sym.identity(ann_index_lowres, name="ann_index_lowres"), \
            mx.sym.identity(index_uv_lowres, name="index_uv_lowres"), \
            mx.sym.identity(u_lowres, name="u_lowres"), \
            mx.sym.identity(v_lowres, name="v_lowres")
