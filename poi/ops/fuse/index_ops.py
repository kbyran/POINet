def heatmap2points(F, heatmap, batch_size, num_joints, heatmap_height, heatmap_width):
    heatmap_reshape = F.reshape(heatmap, shape=(0, 0, -1))
    max_val = F.max(heatmap_reshape, axis=-1, keepdims=True)
    max_flatten_idx = F.argmax(heatmap_reshape, axis=-1, keepdims=True)
    max_y = F.floor(max_flatten_idx / heatmap_width)
    max_x = max_flatten_idx - max_y * heatmap_width
    max_idx = F.concat(max_x, max_y, dim=2)
    valid_val = max_val > 0.0
    valid_val = F.tile(valid_val, reps=(1, 1, 2))
    max_idx = F.where(valid_val, max_idx, F.zeros_like(max_idx))

    max_ym1 = max_y - 1
    max_yp1 = max_y + 1
    max_xm1 = max_x - 1
    max_xp1 = max_x + 1
    max_ym1_clip = F.clip(max_ym1, a_min=0, a_max=heatmap_height - 1)
    max_yp1_clip = F.clip(max_yp1, a_min=0, a_max=heatmap_height - 1)
    max_xm1_clip = F.clip(max_xm1, a_min=0, a_max=heatmap_width - 1)
    max_xp1_clip = F.clip(max_xp1, a_min=0, a_max=heatmap_width - 1)

    heatmap_flatten = F.reshape(heatmap_reshape, shape=(-1,))
    pre_idx = F.reshape(F.arange(start=0, stop=batch_size * num_joints, step=1),
                        (batch_size, num_joints, 1))
    max_val_y_xm1 = F.take(
        a=heatmap_flatten,
        indices=pre_idx * heatmap_width * heatmap_height + max_y * heatmap_width + max_xm1_clip,
        axis=0
    )
    max_val_y_xm1 = F.reshape(max_val_y_xm1, (batch_size, num_joints, 1))
    max_val_y_xp1 = F.take(
        a=heatmap_flatten,
        indices=pre_idx * heatmap_width * heatmap_height + max_y * heatmap_width + max_xp1_clip,
        axis=0
    )
    max_val_y_xp1 = F.reshape(max_val_y_xp1, (batch_size, num_joints, 1))
    max_val_ym1_x = F.take(
        a=heatmap_flatten,
        indices=pre_idx * heatmap_width * heatmap_height + max_ym1_clip * heatmap_width + max_x,
        axis=0
    )
    max_val_ym1_x = F.reshape(max_val_ym1_x, (batch_size, num_joints, 1))
    max_val_yp1_x = F.take(
        a=heatmap_flatten,
        indices=pre_idx * heatmap_width * heatmap_height + max_yp1_clip * heatmap_width + max_x,
        axis=0
    )
    max_val_yp1_x = F.reshape(max_val_yp1_x, (batch_size, num_joints, 1))

    diff = F.concat(max_val_y_xp1 - max_val_y_xm1, max_val_yp1_x - max_val_ym1_x, dim=2)
    valid_y_idx = F.broadcast_logical_and(max_ym1 > 0, max_yp1 < (heatmap_height - 1))
    valid_x_idx = F.broadcast_logical_and(max_xm1 > 0, max_xp1 < (heatmap_width - 1))
    valid_idx = F.broadcast_logical_and(valid_y_idx, valid_x_idx)
    valid_idx = F.tile(valid_idx, reps=(1, 1, 2))
    sign = F.where(diff < 0, -F.ones_like(diff), F.where(diff > 0, F.ones_like(diff), diff))
    max_idx = F.where(valid_idx, max_idx + sign * 0.25, max_idx)
    return max_idx, max_val


def resize_uv(F, ann_index_lowres, index_uv_lowres, u_lowres, v_lowres, height, width):
    ann_index_lowres = F.expand_dims(ann_index_lowres, axis=0)
    ann_index = F.contrib.BilinearResize2D(ann_index_lowres, height=height, width=width)
    index_uv_lowres = F.expand_dims(index_uv_lowres, axis=0)
    index_uv = F.contrib.BilinearResize2D(index_uv_lowres, height=height, width=width)
    u_lowres = F.expand_dims(u_lowres, axis=0)
    u = F.contrib.BilinearResize2D(u_lowres, height=height, width=width)
    v_lowres = F.expand_dims(v_lowres, axis=0)
    v = F.contrib.BilinearResize2D(v_lowres, height=height, width=width)

    ann_index = F.argmax(ann_index, axis=1)
    index_uv = F.argmax(index_uv, axis=1)
    index_uv = (ann_index > 0) * index_uv

    u_transpose = F.transpose(u, axes=(0, 2, 3, 1))
    v_transpose = F.transpose(v, axes=(0, 2, 3, 1))
    u = F.pick(u_transpose, index_uv)
    v = F.pick(v_transpose, index_uv)

    uv_stack = F.concat(index_uv, u, v, dim=0)

    return uv_stack
