def _eye_like(F, vector):
    ones_array = F.ones_like(vector)
    eye_array = F.diag(ones_array)
    return eye_array


def _pos_mask(F, labels):
    indices_equal = _eye_like(F, labels)
    indices_not_equal = F.logical_not(indices_equal)
    labels_equal = F.broadcast_equal(F.expand_dims(labels, 0), F.expand_dims(labels, 1))
    pos_mask = F.broadcast_logical_and(indices_not_equal, labels_equal)
    return pos_mask


def _neg_mask(F, labels):
    labels_not_equal = F.broadcast_not_equal(F.expand_dims(labels, 0), F.expand_dims(labels, 1))
    return labels_not_equal


def _pairwise_distances(F, embeddings):
    square = F.sum(F.square(data=embeddings), axis=1, keepdims=True)
    dot_product = F.dot(lhs=embeddings, rhs=embeddings, transpose_a=False, transpose_b=True)
    distances = F.broadcast_add(square, F.transpose(square)) - dot_product * 2.0
    distances = F.sqrt(F.maximum(distances, 1e-12))  # TODO: sqrt
    return distances


def batch_hard_triplet_loss(F, embeddings, labels, margin=0.2):
    import numpy as np
    pairwise_dist = _pairwise_distances(F, embeddings)
    pos_mask = _pos_mask(F, labels)
    neg_mask = _neg_mask(F, labels)
    inf_array = F.ones_like(pairwise_dist) * np.inf
    anchor_pos_dist = F.where(pos_mask, pairwise_dist, -inf_array)
    max_anchor_pos_dist = F.topk(anchor_pos_dist, ret_typ="value", axis=1, is_ascend=False)
    anchor_neg_dist = F.where(neg_mask, pairwise_dist, inf_array)
    min_anchor_neg_dist = F.topk(anchor_neg_dist, ret_typ="value", axis=1, is_ascend=True)
    if margin == "soft":
        # triplet_loss = F.log10(F.exp(max_anchor_pos_dist - min_anchor_neg_dist) + 1.0)  # log10
        triplet_loss = F.Activation(
            max_anchor_pos_dist - min_anchor_neg_dist,
            act_type="softrelu"
        )  # log, F.Activation is more stable
    else:
        triplet_loss = F.relu(max_anchor_pos_dist - min_anchor_neg_dist + margin)
    # triplet_loss = F.mean(triplet_loss, axis=0, keepdims=True)
    return triplet_loss


def _c_distance(F, embeddings, centers):
    # embeddings (batch_size, feat_dim)
    # centers (num_classes, feat_dim)
    # out (batch_size, num_classes)
    sq_embbeddings = F.sum(F.square(data=embeddings), axis=1, keepdims=True)
    centers_t = F.transpose(centers, axes=(1, 0))
    sq_centers_t = F.sum(F.square(data=centers_t), axis=0, keepdims=True)
    dot_product = F.dot(lhs=embeddings, rhs=centers_t, transpose_a=False, transpose_b=False)
    distances = F.broadcast_add(sq_embbeddings, sq_centers_t) - dot_product * 2.0
    distances = F.minimum(F.maximum(distances, 1e-12), 1e+12)  # TODO: sqrt
    return distances


def center_loss(F, embeddings, centers, labels):
    distances = _c_distance(F, embeddings, centers)
    center_labels = F.contrib.arange_like(centers, axis=0, start=0)
    labels_equal = F.broadcast_equal(F.expand_dims(center_labels, 0), F.expand_dims(labels, 1))
    center_loss = distances * labels_equal
    center_loss = F.mean(center_loss, axis=1, keepdims=True)
    # center_loss = F.sum(center_loss, axis=1, keepdims=True)
    # center_loss = F.mean(F.sum(center_loss, axis=1), axis=0, keepdims=True)
    return center_loss


if __name__ == "__main__":
    import mxnet as mx
    F = mx.ndarray
    labels = F.arange(16)
    print(_eye_like(F, labels))
    labels = F.repeat(F.arange(4), repeats=2)
    print(_pos_mask(F, labels))
    print(_neg_mask(F, labels))
    embeddings = F.reshape(F.arange(16), (2, 8))
    print(_pairwise_distances(F, embeddings))
    # labels = F.repeat(F.arange(2), repeats=2)
    labels = F.array([1, 3, 3, 3])
    embeddings = F.reshape(F.arange(32), (4, 8))
    print(batch_hard_triplet_loss(F, embeddings, labels, margin="soft"))
    centers = F.reshape(F.arange(48), (6, 8))
    print(center_loss(F, embeddings, centers, labels))
