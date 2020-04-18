import mxnet as mx


class Builder(object):
    def __init__(self):
        pass

    @staticmethod
    def get_train_symbol(backbone, neck, head):
        data = mx.sym.var("data")
        labels = mx.sym.var("labels")
        feat = backbone.get_feature(data)
        feat = neck.get_feature(feat)
        loss = head.get_loss(feat, labels)

        return loss

    @staticmethod
    def get_test_symbol(backbone, neck, head):
        data = mx.sym.var("data")
        im_id = mx.sym.var("im_id")
        rec_id = mx.sym.var("rec_id")

        feat = backbone.get_feature(data)
        feat = neck.get_feature(feat)
        feat = head.get_output(feat)

        return mx.sym.Group([rec_id, im_id, feat])

    @staticmethod
    def get_export_symbol(backbone, neck, head):
        data = mx.sym.var("data")

        feat = backbone.get_feature(data)
        feat = neck.get_feature(feat)
        feat = head.get_output(feat)

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

    def get_logits(self, feature):
        p = self.p
        dropout = p.xent_loss.dropout
        dropout_p = p.xent_loss.dropout_p
        num_classes = p.xent_loss.num_classes
        if dropout:
            feature = mx.sym.Dropout(data=feature, p=dropout_p)
        fc1 = mx.sym.FullyConnected(
            data=feature,
            weight=self.cls_weight,
            num_hidden=num_classes,
            no_bias=True,
            name="fc1"
        )
        return fc1

    def get_xent_loss(self, logits, labels):
        p = self.p
        num_classes = p.xent_loss.num_classes
        pos_ratio = p.xent_loss.pos_ratio
        grad_scale = p.xent_loss.grad_scale

        pos_weight = mx.sym.var(
            "pos_weight",
            shape=(num_classes,),
            init=mx.init.Constant((1. - mx.nd.array(pos_ratio)) / mx.nd.array(pos_ratio)),
            lr_mult=0,
            wd_mult=0
        )
        # weighted bce loss
        # pos_weight = mx.sym.zeros_like(labels)
        log_weight = 1.0 + mx.sym.broadcast_mul(pos_weight - 1.0, labels)
        loss = logits - logits * labels + log_weight * \
            (mx.sym.Activation(-mx.sym.abs(logits), act_type='softrelu') + mx.sym.relu(-logits))
        xent_loss = mx.sym.MakeLoss(loss, grad_scale=grad_scale, name="bce_loss")

        return xent_loss, mx.sym.BlockGrad(logits, name="fc1")

    def get_loss(self, feature, labels):
        p = self.p
        xent_feature = p.xent_loss.feature

        loss = list()
        logits = self.get_logits(feature[xent_feature])
        xent_loss, fc_output = self.get_xent_loss(logits, labels)
        loss += [xent_loss, fc_output]

        return mx.sym.Group(loss)

    def get_output(self, feature):
        p = self.p
        xent_feature = p.xent_loss.feature
        logits = self.get_logits(feature[xent_feature])
        outputs = mx.sym.sigmoid(logits)

        return outputs
