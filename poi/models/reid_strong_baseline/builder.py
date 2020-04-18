import mxnet as mx


class Builder(object):
    def __init__(self):
        pass

    @staticmethod
    def get_train_symbol(backbone, neck, head):
        data = mx.sym.var("data")
        label = mx.sym.var("label")
        feat = backbone.get_feature(data)
        feat = neck.get_feature(feat)
        loss = head.get_loss(feat, label)

        return loss

    @staticmethod
    def get_test_symbol(backbone, neck, head, fliptest=False):
        data = mx.sym.var("data")
        if fliptest:
            data_flip = mx.sym.flip(data, axis=3, name="data_flip")
            data = mx.sym.concat(data, data_flip, dim=0)
        im_id = mx.sym.var("im_id")
        rec_id = mx.sym.var("rec_id")

        feat = backbone.get_feature(data)
        feat = neck.get_feature(feat)
        feat = head.get_feature(feat)

        if fliptest:
            feat = mx.sym.reshape(feat, shape=(-4, 2, -1, -2))
            feat = mx.sym.mean(feat, axis=0)

        return mx.sym.Group([rec_id, im_id, feat])

    @staticmethod
    def get_export_symbol(backbone, neck, head, fliptest=False):
        data = mx.sym.var("data")
        if fliptest:
            data_flip = mx.sym.flip(data, axis=3, name="data_flip")
            data = mx.sym.concat(data, data_flip, dim=0)

        feat = backbone.get_feature(data)
        feat = neck.get_feature(feat)
        feat = head.get_feature(feat)

        if fliptest:
            feat = mx.sym.reshape(feat, shape=(-4, 2, -1, -2))
            feat = mx.sym.mean(feat, axis=0)

        return feat


class BasicNeck(object):
    def __init__(self, pNeck):
        self.p = pNeck
        self.feat = None

    def get_feature(self, feature):
        feat_global = mx.sym.Pooling(data=feature, kernel=(16, 8), pool_type="avg",
                                     global_pool=True, name="feat_global")
        feat = mx.sym.flatten(data=feat_global, name="feat_before")
        self.feat = dict(
            pool_branch=feat,
        )

        return self.feat


class BNNeck(object):
    def __init__(self, pNeck):
        self.p = pNeck

        self.neck_bn_gamma = mx.sym.var("neck_bn_gamma", init=mx.init.One())
        self.neck_bn_beta = mx.sym.var(
            "neck_bn_beta", init=mx.init.Zero(), lr_mult=0.0, wd_mult=0.0)
        # self.neck_bn_moving_mean = mx.sym.var("neck_bn_moving_mean", init=mx.init.Zero())
        # self.neck_bn_moving_var = mx.sym.var("neck_bn_moving_var", init=mx.init.Zero())

        self.feat = None

    def get_feature(self, feature):
        feat_global = mx.sym.Pooling(data=feature, kernel=(16, 8), pool_type="avg",
                                     global_pool=True, name="feat_global")
        feat_before = mx.sym.flatten(data=feat_global, name="feat_before")
        feat_after = mx.sym.BatchNorm(
            data=feat_before,
            gamma=self.neck_bn_gamma,
            beta=self.neck_bn_beta,
            # moving_mean=self.neck_bn_moving_mean,
            # moving_var=self.neck_bn_moving_var,
            eps=1e-5 + 1e-10,
            momentum=0.9,
            fix_gamma=False,
            use_global_stats=False,
            name="neck_bn"
        )
        self.feat = dict(
            pool_branch=feat_before,
            bn_branch=feat_after
        )

        return self.feat


class MultiHead(object):
    def __init__(self, pHead):
        self.p = pHead

        # classification weight
        self.cls_weight = mx.sym.var("cls_weight", init=mx.init.Normal(sigma=0.001))
        # centers
        self.centers = mx.sym.var("centers", shape=(751, 2048), init=mx.init.Normal(sigma=0.1))

    def get_xent_loss(self, feature, label):
        p = self.p
        num_classes = p.xent_loss.num_classes
        grad_scale = p.xent_loss.grad_scale
        smooth_alpha = p.xent_loss.smooth_alpha

        fc1 = mx.sym.FullyConnected(
            data=feature,
            weight=self.cls_weight,
            num_hidden=num_classes,
            no_bias=True,
            name="fc1"
        )
        xent_loss = mx.sym.SoftmaxOutput(
            data=fc1,
            label=label,
            grad_scale=grad_scale,
            normalization="batch",
            smooth_alpha=smooth_alpha,
            name="softmax"
        )
        # pred = mx.sym.log_softmax(fc1, -1)
        # loss = - mx.sym.pick(pred, label, axis=-1, keepdims=True)
        # loss = mx.sym.mean(loss, axis=-1)
        # xent_loss = mx.sym.MakeLoss(loss, grad_scale=grad_scale, name="softmax")

        return xent_loss

    def get_triplet_loss(self, feature, label):
        from poi.ops.fuse.metric_ops import batch_hard_triplet_loss
        p = self.p
        margin = p.triplet_loss.margin
        grad_scale = p.triplet_loss.grad_scale

        triplet_loss = batch_hard_triplet_loss(mx.sym, feature, label, margin=margin)
        triplet_loss = mx.sym.MakeLoss(
            triplet_loss,
            grad_scale=grad_scale,
            normalization="batch",
            name="triplet_loss"
        )

        return triplet_loss

    def get_center_loss(self, feature, label):
        from poi.ops.fuse.metric_ops import center_loss
        p = self.p
        grad_scale = p.center_loss.grad_scale

        center_loss = center_loss(mx.sym, feature, self.centers, label)
        center_loss = mx.sym.MakeLoss(
            center_loss,
            grad_scale=grad_scale,
            normalization="batch",
            name="center_loss"
        )

        return center_loss

    def get_loss(self, feature, label):
        p = self.p
        xent_enable = p.xent_loss.is_enable
        xent_feature = p.xent_loss.feature
        triplet_enable = p.triplet_loss.is_enable
        triplet_feature = p.triplet_loss.feature
        center_enable = p.center_loss.is_enable
        center_feature = p.center_loss.feature

        loss = list()
        if xent_enable:
            xent_loss = self.get_xent_loss(feature[xent_feature], label)
            loss.append(xent_loss)
        if triplet_enable:
            triplet_loss = self.get_triplet_loss(feature[triplet_feature], label)
            loss.append(triplet_loss)
        if center_enable:
            center_loss = self.get_center_loss(feature[center_feature], label)
            loss.append(center_loss)

        assert len(loss) > 0, "There should be at least one loss."
        if len(loss) == 0:
            return loss[0]
        else:
            return mx.sym.Group(loss)

    def get_feature(self, feature):
        p = self.p
        output_feature = p.get_feature.feature

        return feature[output_feature]
