import logging
import onnx
import numpy as np
from mxnet.contrib.onnx.mx2onnx._op_translations import (
    parse_helper, get_inputs, get_boolean_attribute_value, create_basic_op_node)
from mxnet.contrib.onnx.mx2onnx.export_onnx import MXNetGraph as mx_op


@mx_op.register("FullyConnected")
def convert_fully_connected(node, **kwargs):
    """Map MXNet's FullyConnected operator attributes to onnx's Gemm operator
    and return the created node.
    """
    name, input_nodes, attrs = get_inputs(node, kwargs)

    initializer = kwargs["initializer"]

    no_bias = get_boolean_attribute_value(attrs, "no_bias")

    fcnode = []

    op_name = "flatten_" + str(kwargs["idx"])
    flatten_node = onnx.helper.make_node(
        'Flatten',
        inputs=[input_nodes[0]],
        outputs=[op_name],
        name=op_name
    )

    input_nodes[0] = op_name
    fcnode.append(flatten_node)

    if no_bias:
        data_type = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(attrs.get('dtype', 'float32'))]
        bias_name = "bias" + str(kwargs["idx"])
        tensor_node = onnx.helper.make_tensor_value_info(bias_name, data_type, (1,))
        initializer.append(
            onnx.helper.make_tensor(
                name=bias_name,
                data_type=data_type,
                dims=(1,),
                vals=[0],
                raw=False,
            )
        )
        input_nodes.append(bias_name)
        fcnode.append(tensor_node)

    node = onnx.helper.make_node(
        "Gemm",
        input_nodes,  # input (A, B, C) - C can be in place
        [name],  # output
        alpha=1.0,
        beta=1.0,
        transA=False,
        transB=True,
        name=name
    )

    fcnode.append(node)

    return fcnode


@mx_op.register("BatchNorm")
def convert_batchnorm(node, **kwargs):
    """Map MXNet's BatchNorm operator attributes to onnx's BatchNormalization operator
    and return the created node.
    """
    name, input_nodes, attrs = get_inputs(node, kwargs)

    momentum = float(attrs.get("momentum", 0.9))
    eps = float(attrs.get("eps", 0.001))

    bn_node = onnx.helper.make_node(
        "BatchNormalization",
        input_nodes,
        [name],
        name=name,
        epsilon=eps,
        momentum=momentum,
    )
    return [bn_node]


@mx_op.register("Pooling")
def convert_pooling(node, **kwargs):
    """Map MXNet's Pooling operator attributes to onnx's
    MaxPool/AveragePool/GlobalMaxPool/GlobalAveragePool operators
    based on the input node's attributes and return the created node.
    """
    name, input_nodes, attrs = get_inputs(node, kwargs)

    kernel = eval(attrs["kernel"])
    pool_type = attrs["pool_type"] if attrs.get("pool_type") else "max"
    stride = eval(attrs["stride"]) if attrs.get("stride") else (1, 1)
    global_pool = get_boolean_attribute_value(attrs, "global_pool")
    p_value = attrs.get('p_value', 'None')

    pooling_convention = attrs.get('pooling_convention', 'valid')

    if pooling_convention == 'full':
        pooling_warning = "Pooling: ONNX currently doesn't support pooling_convention. " \
                          "This might lead to shape or accuracy issues. " \
                          "https://github.com/onnx/onnx/issues/549"

        logging.warning(pooling_warning)

    # ceil_mode = 1 if attrs.get("pooling_convention", "valid") == "full" else 0

    pad_dims = list(parse_helper(attrs, "pad", [0, 0]))
    pad_dims = pad_dims + pad_dims
    pool_types = {"max": "MaxPool", "avg": "AveragePool", "lp": "LpPool"}
    global_pool_types = {"max": "GlobalMaxPool", "avg": "GlobalAveragePool",
                         "lp": "GlobalLpPool"}
    count_include_pad = 1 if attrs.get("count_include_pad", "True") in ["True", "1"] else 0

    if pool_type == 'lp' and p_value == 'None':
        raise AttributeError('ONNX requires a p value for LpPool and GlobalLpPool')

    if global_pool:
        if pool_type == 'lp':
            node = onnx.helper.make_node(
                global_pool_types[pool_type],
                input_nodes,  # input
                [name],
                p=int(p_value),
                name=name
            )
        else:
            node = onnx.helper.make_node(
                global_pool_types[pool_type],
                input_nodes,  # input
                [name],
                name=name
            )
    else:
        if pool_type == 'lp':
            node = onnx.helper.make_node(
                pool_types[pool_type],
                input_nodes,  # input
                [name],
                p=int(p_value),
                kernel_shape=kernel,
                pads=pad_dims,
                strides=stride,
                name=name
            )
        elif pool_type == "avg":
            node = onnx.helper.make_node(
                pool_types[pool_type],
                input_nodes,  # input
                [name],
                count_include_pad=count_include_pad,
                kernel_shape=kernel,
                pads=pad_dims,
                strides=stride,
                # ceil_mode=ceil_mode,
                name=name
            )
        else:
            node = onnx.helper.make_node(
                pool_types[pool_type],
                input_nodes,  # input
                [name],
                kernel_shape=kernel,
                pads=pad_dims,
                strides=stride,
                name=name
            )

    return [node]


@mx_op.register("clip")
def convert_clip(node, **kwargs):
    """Map MXNet's Clip operator attributes to onnx's Clip operator
    and return the created node.
    """
    name, input_nodes, attrs = get_inputs(node, kwargs)

    initializer = kwargs["initializer"]
    data_type = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype('float32')]
    a_min_name = "a_min" + str(kwargs["idx"])
    a_min_vals = np.array([float(attrs.get('a_min', -np.inf))])
    a_min_node = onnx.helper.make_tensor_value_info(a_min_name, data_type, [])
    initializer.append(
        onnx.helper.make_tensor(
            name=a_min_name,
            data_type=data_type,
            dims=[],
            vals=a_min_vals,
            raw=False
        )
    )
    a_max_name = "a_max" + str(kwargs["idx"])
    a_max_vals = np.array([float(attrs.get('a_max', -np.inf))])
    a_max_node = onnx.helper.make_tensor_value_info(a_max_name, data_type, [])
    initializer.append(
        onnx.helper.make_tensor(
            name=a_max_name,
            data_type=data_type,
            dims=[],
            vals=a_max_vals,
            raw=False
        )
    )

    clip_node = onnx.helper.make_node(
        "Clip",
        [input_nodes[0], a_min_name, a_max_name],
        [name],
        name=name,
    )
    return [a_min_node, a_max_node, clip_node]


@mx_op.register("_arange")
def convert_arange(node, **kwargs):
    """Map MXNet's Arange operator attributes to onnx's Range operator
    and return the created node.
    """
    input_type = kwargs["in_type"]
    name, input_nodes, attrs = get_inputs(node, kwargs)
    initializer = kwargs["initializer"]
    start_value = np.array([float(attrs.get("start", 0))],
                           dtype=onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[input_type])
    start_name = "start_" + str(kwargs["idx"])
    start_node = onnx.helper.make_tensor_value_info(start_name, input_type, [])
    initializer.append(
        onnx.helper.make_tensor(
            name=start_name,
            data_type=input_type,
            dims=[],
            vals=start_value,
            raw=False,
        )
    )

    limit_value = np.array([float(attrs.get("stop", 1))],
                           dtype=onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[input_type])
    limit_name = "limit_" + str(kwargs["idx"])
    limit_node = onnx.helper.make_tensor_value_info(limit_name, input_type, [])
    initializer.append(
        onnx.helper.make_tensor(
            name=limit_name,
            data_type=input_type,
            dims=[],
            vals=limit_value,
            raw=False,
        )
    )

    delta_value = np.array([float(attrs.get("step", 1))],
                           dtype=onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[input_type])
    delta_name = "delta_" + str(kwargs["idx"])
    delta_node = onnx.helper.make_tensor_value_info(delta_name, input_type, [])
    initializer.append(
        onnx.helper.make_tensor(
            name=delta_name,
            data_type=input_type,
            dims=[],
            vals=delta_value,
            raw=False,
        )
    )
    range_node = onnx.helper.make_node(
        'Range',
        [start_name, limit_name, delta_name],
        [name],
        name=name
    )
    return [start_node, limit_node, delta_node, range_node]


@mx_op.register("_greater_scalar")
def convert_greater_scalar(node, **kwargs):
    """Map MXNet's _greater_scalar operator attributes to onnx's Greater operator.
    Creates a new node for the input scalar value, adds it to the initializer
    and return multiple created nodes.
    """
    name, input_nodes, attrs = get_inputs(node, kwargs)
    input_type = kwargs["in_type"]
    scalar_value = np.array([attrs.get("scalar", 1)],
                            dtype=onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[input_type])

    initializer = kwargs["initializer"]
    dims = np.shape(scalar_value)

    scalar_op_name = "scalar_op" + str(kwargs["idx"])
    tensor_node = onnx.helper.make_tensor_value_info(scalar_op_name, input_type, dims)

    initializer.append(
        onnx.helper.make_tensor(
            name=scalar_op_name,
            data_type=input_type,
            dims=dims,
            vals=scalar_value,
            raw=False,
        )
    )

    greater_node = onnx.helper.make_node(
        "Greater",
        [input_nodes[0], scalar_op_name],
        [name],
        name=name
    )

    dtype = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[kwargs["in_type"]]
    if dtype == 'float32':
        dtype = 'float'
    elif dtype == 'float64':
        dtype = 'double'
    cast_name = 'cast_' + name
    cast_node = onnx.helper.make_node(
        "Cast",
        [name],
        [cast_name],
        to=getattr(onnx.TensorProto, dtype.upper()),
        name=cast_name,
    )

    return [tensor_node, greater_node, cast_node]


@mx_op.register("_lesser_scalar")
def convert_lesser_scalar(node, **kwargs):
    """Map MXNet's _lesser_scalar operator attributes to onnx's Less operator.
    Creates a new node for the input scalar value, adds it to the initializer
    and return multiple created nodes.
    """
    name, input_nodes, attrs = get_inputs(node, kwargs)
    input_type = kwargs["in_type"]
    scalar_value = np.array([attrs.get("scalar", 1)],
                            dtype=onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[input_type])

    initializer = kwargs["initializer"]
    dims = np.shape(scalar_value)

    scalar_op_name = "scalar_op" + str(kwargs["idx"])
    tensor_node = onnx.helper.make_tensor_value_info(scalar_op_name, input_type, dims)

    initializer.append(
        onnx.helper.make_tensor(
            name=scalar_op_name,
            data_type=input_type,
            dims=dims,
            vals=scalar_value,
            raw=False,
        )
    )

    less_node = onnx.helper.make_node(
        "Less",
        [input_nodes[0], scalar_op_name],
        [name],
        name=name
    )

    dtype = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[kwargs["in_type"]]
    if dtype == 'float32':
        dtype = 'float'
    elif dtype == 'float64':
        dtype = 'double'
    cast_name = 'cast_' + name
    cast_node = onnx.helper.make_node(
        "Cast",
        [name],
        [cast_name],
        to=getattr(onnx.TensorProto, dtype.upper()),
        name=cast_name,
    )

    return [tensor_node, less_node, cast_node]


@mx_op.register("zeros_like")
def convert_zeros_like(node, **kwargs):
    """Map MXNet's zeros_like operator attributes to onnx's Shape and ConstantOfShape operator
    and return the created node.
    """
    name, input_nodes, attrs = get_inputs(node, kwargs)
    zlnode = []
    op_name = "zeros_like_shape" + str(kwargs["idx"])
    shape_node = onnx.helper.make_node(
        'Shape',
        inputs=[input_nodes[0]],
        outputs=[op_name],
        name=op_name
    )

    input_nodes[0] = op_name
    zlnode.append(shape_node)

    tensor_node = onnx.helper.make_tensor(
        "zeros_like_value" + str(kwargs["idx"]), onnx.TensorProto.FLOAT, [1], [0])
    shape_like_node = onnx.helper.make_node(
        'ConstantOfShape',
        inputs=[input_nodes[0]],
        outputs=[name],
        value=tensor_node,
        name=name
    )
    zlnode.append(shape_like_node)

    return zlnode


@mx_op.register("ones_like")
def convert_ones_like(node, **kwargs):
    """Map MXNet's ones_like operator attributes to onnx's Shape and ConstantOfShape operator
    and return the created node.
    """
    name, input_nodes, attrs = get_inputs(node, kwargs)
    olnode = []
    op_name = "ones_like_shape" + str(kwargs["idx"])
    shape_node = onnx.helper.make_node(
        'Shape',
        inputs=[input_nodes[0]],
        outputs=[op_name],
        name=op_name
    )

    input_nodes[0] = op_name
    olnode.append(shape_node)

    tensor_node = onnx.helper.make_tensor(
        "zeros_like_value" + str(kwargs["idx"]), onnx.TensorProto.FLOAT, [1], [1])
    shape_like_node = onnx.helper.make_node(
        'ConstantOfShape',
        inputs=[input_nodes[0]],
        outputs=[name],
        value=tensor_node,
        name=name
    )
    olnode.append(shape_like_node)

    return olnode


@mx_op.register("where")
def convert_where(node, **kwargs):
    """Map MXNet's where operator attributes to onnx's Where operator
    and return the created node.
    """
    name, input_nodes, attrs = get_inputs(node, kwargs)
    dtype = 'bool'
    cast_name = 'cast_' + name
    cast_node = onnx.helper.make_node(
        "Cast",
        [input_nodes[0]],
        [cast_name],
        to=getattr(onnx.TensorProto, dtype.upper()),
        name=cast_name,
    )
    node = onnx.helper.make_node(
        "Where",
        [cast_name, input_nodes[1], input_nodes[2]],
        [name],
        name=name
    )
    return [cast_node, node]


@mx_op.register("pick")
def convert_pick(node, **kwargs):
    """Map MXNet's pick operator attributes to onnx's GatherElements operator
    and return the created node.
    """
    name, input_nodes, attrs = get_inputs(node, kwargs)
    dtype = 'int32'
    cast_name = 'cast_' + name
    cast_node = onnx.helper.make_node(
        "Cast",
        [input_nodes[1]],
        [cast_name],
        to=getattr(onnx.TensorProto, dtype.upper()),
        name=cast_name,
    )

    node = onnx.helper.make_node(
        "GatherElements",
        [input_nodes[0], cast_name],
        [name],
        axis=int(attrs.get("axis", -1)),
        name=name
    )
    return [cast_node, node]


@mx_op.register("take")
def convert_take(node, **kwargs):
    """Map MXNet's take operator attributes to onnx's Gather operator
    and return the created node.
    """
    name, input_nodes, attrs = get_inputs(node, kwargs)
    dtype = 'int32'
    cast_name = 'cast_' + name
    cast_node = onnx.helper.make_node(
        "Cast",
        [input_nodes[1]],
        [cast_name],
        to=getattr(onnx.TensorProto, dtype.upper()),
        name=cast_name,
    )

    node = onnx.helper.make_node(
        "Gather",
        [input_nodes[0], cast_name],
        [name],
        axis=int(attrs.get("axis", -1)),
        name=name
    )
    return [cast_node, node]


@mx_op.register("sign")
def convert_sign(node, **kwargs):
    """Map MXNet's sign operator attributes to onnx's Sign operator
    and return the created node.
    """
    return create_basic_op_node('Sign', node, kwargs)


@mx_op.register("argmax")
def convert_argmax(node, **kwargs):
    """Map MXNet's argmax operator attributes to onnx's ArgMax operator
    and return the created node.
    """
    name, input_nodes, attrs = get_inputs(node, kwargs)

    axis = int(attrs.get("axis"))
    keepdims = get_boolean_attribute_value(attrs, "keepdims")

    argmax_node = onnx.helper.make_node(
        'ArgMax',
        inputs=input_nodes,
        axis=axis,
        keepdims=keepdims,
        outputs=[name],
        name=name
    )

    dtype = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[kwargs["in_type"]]
    if dtype == 'float32':
        dtype = 'float'
    elif dtype == 'float64':
        dtype = 'double'
    cast_name = 'cast_' + name
    cast_node = onnx.helper.make_node(
        "Cast",
        [name],
        [cast_name],
        to=getattr(onnx.TensorProto, dtype.upper()),
        name=cast_name,
    )

    return [argmax_node, cast_node]


@mx_op.register("broadcast_logical_and")
def convert_broadcast_logical_and(node, **kwargs):
    """Map MXNet's broadcast_logical_and operator attributes to onnx's And operator
    and return the created node.
    """
    name, input_nodes, _ = get_inputs(node, kwargs)
    nodes = []
    cast_names = []
    for node in input_nodes:
        cast_name = 'cast_' + node
        bool_node = onnx.helper.make_node(
            'Cast',
            [node],
            [cast_name],
            to=getattr(onnx.TensorProto, "BOOL"),
            name=cast_name
        )
        nodes.append(bool_node)
        cast_names.append(cast_name)
    node = onnx.helper.make_node(
        'And',
        cast_names,
        [name],
        name=name
    )
    nodes.append(node)
    dtype = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[kwargs["in_type"]]
    if dtype == 'float32':
        dtype = 'float'
    elif dtype == 'float64':
        dtype = 'double'
    cast_name = 'cast_' + name
    cast_node = onnx.helper.make_node(
        "Cast",
        [name],
        [cast_name],
        to=getattr(onnx.TensorProto, dtype.upper()),
        name=cast_name,
    )
    nodes.append(cast_node)

    return nodes


from mxnet.contrib import onnx as onnx_mxnet
onnx_mxnet.mx2onnx._op_translations.convert_fully_connected = convert_fully_connected
onnx_mxnet.mx2onnx._op_translations.convert_batchnorm = convert_batchnorm
onnx_mxnet.mx2onnx._op_translations.convert_pooling = convert_pooling
onnx_mxnet.mx2onnx._op_translations.convert_clip = convert_clip
onnx_mxnet.mx2onnx._op_translations.convert_arange = convert_arange
onnx_mxnet.mx2onnx._op_translations.convert_greater_scalar = convert_greater_scalar
onnx_mxnet.mx2onnx._op_translations.convert_lesser_scalar = convert_lesser_scalar
onnx_mxnet.mx2onnx._op_translations.convert_zeros_like = convert_zeros_like
onnx_mxnet.mx2onnx._op_translations.convert_ones_like = convert_ones_like
onnx_mxnet.mx2onnx._op_translations.convert_where = convert_where
onnx_mxnet.mx2onnx._op_translations.convert_pick = convert_pick
onnx_mxnet.mx2onnx._op_translations.convert_take = convert_take
onnx_mxnet.mx2onnx._op_translations.convert_sign = convert_sign
onnx_mxnet.mx2onnx._op_translations.convert_argmax = convert_argmax
onnx_mxnet.mx2onnx._op_translations.convert_broadcast_logical_and = convert_broadcast_logical_and
