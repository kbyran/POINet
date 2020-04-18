import numpy as np
import mxnet as mx
from poi.ops.fuse.metric_ops import _c_distance


def c_distance_opt(q_feats, g_feats, ctx=mx.cpu(), max_size=10000, dtype=np.float32,
                   normalize=True):
    num_q = q_feats.shape[0]
    num_g = g_feats.shape[0]
    dist = np.ones((num_q, num_g), dtype=dtype)
    for i in range(0, num_q, max_size):
        start_i = i
        end_i = min(i + max_size, num_q)
        for j in range(0, num_g, max_size):
            start_j = j
            end_j = min(j + max_size, num_g)
            print(start_i, end_i, start_j, end_j)
            q_feat_nd = mx.nd.array(q_feats[start_i: end_i], ctx=ctx)
            g_feat_nd = mx.nd.array(g_feats[start_j: end_j], ctx=ctx)
            if normalize:
                q_feat_nd = mx.nd.L2Normalization(q_feat_nd, mode="instance")
                g_feat_nd = mx.nd.L2Normalization(g_feat_nd, mode="instance")
            dist[start_i: end_i, start_j: end_j] = \
                _c_distance(mx.nd, q_feat_nd, g_feat_nd).asnumpy().astype(dtype)

    return dist


def re_ranking(probFea, galFea, k1, k2, lambda_value, local_distmat=None, only_local=False,
               ctx=mx.cpu()):
    # if feature vector is numpy, you should use 'torch.tensor' transform it to tensor
    query_num = len(probFea)
    all_num = query_num + len(galFea)
    if only_local:
        original_dist = local_distmat
    else:
        probFea = np.array(probFea, dtype=np.float32)
        galFea = np.array(galFea, dtype=np.float32)
        # print(probFea.shape, galFea.shape)
        feats = np.concatenate((probFea, galFea), axis=0)
        # print(feats.shape)
        print('using GPU to compute original distance')
        original_dist = c_distance_opt(feats, feats, dtype=np.float16, ctx=ctx)
        # print(original_dist.shape)
        del feats
        # probFea_nd = mx.nd.array(probFea, ctx=ctx)
        # probFea_nd = mx.nd.L2Normalization(probFea_nd, mode="instance")
        # galFea_nd = mx.nd.array(galFea, ctx=ctx)
        # galFea_nd = mx.nd.L2Normalization(galFea_nd, mode="instance")
        # print(probFea_nd.shape, galFea_nd.shape)
        # feat_nd = mx.nd.concat(probFea_nd, galFea_nd, dim=0)
        # print(feat_nd.shape)
        # distmat_nd = _c_distance(mx.nd, feat_nd, feat_nd)
        # original_dist = distmat_nd.asnumpy()
        # feat = torch.cat([probFea, galFea])
        # distmat = torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(all_num, all_num) + \
        #     torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(all_num, all_num).t()
        # distmat.addmm_(1, -2, feat, feat.t())
        # original_dist = distmat.cpu().numpy()
        # del feat_nd
        if local_distmat is not None:
            original_dist = original_dist + local_distmat
        # print("dist: ", original_dist.shape)
    gallery_num = original_dist.shape[0]
    # print("start transpose")
    original_dist = np.transpose(original_dist / np.max(original_dist, axis=0))
    # V = np.zeros_like(original_dist, dtype=np.float16)
    # print("start V")
    V = np.zeros((all_num, all_num), dtype=np.float16)
    # print("end V")
    max_rank = max(k1 + 1, k2) + 1
    initial_rank = np.argpartition(original_dist,
                                   list(range(1, max_rank)))[:, :max_rank].astype(np.int32)
    # print(initial_rank.shape)
    # print("end rank")

    print('starting re_ranking')
    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i, :k1 + 1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate, :int(np.around(k1 / 2)) + 1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,
                                                            :int(np.around(k1 / 2)) + 1]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(
                    candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(
                    k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = weight / np.sum(weight)
    original_dist = original_dist[:query_num, ]
    if k2 != 1:
        # print("start v_qe")
        V_qe = np.zeros_like(V, dtype=np.float16)
        # print("end v_qe")
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(gallery_num):
        invIndex.append(np.where(V[:, i] != 0)[0])

    # print("start jaccard_dist")
    jaccard_dist = np.zeros_like(original_dist, dtype=np.float16)
    # print("end jaccard_dist")

    for i in range(query_num):
        temp_min = np.zeros(shape=[1, gallery_num], dtype=np.float16)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + \
                np.minimum(V[i, indNonZero[j]], V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)

    final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num, query_num:]
    return final_dist
