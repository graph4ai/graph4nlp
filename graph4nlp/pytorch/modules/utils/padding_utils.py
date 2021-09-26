import numpy as np


def make_batches(size, batch_size):
    nb_batch = int(np.ceil(size / float(batch_size)))
    return [(i * batch_size, min(size, (i + 1) * batch_size)) for i in range(0, nb_batch)]


def pad_2d_vals_no_size(in_vals, dtype=np.int32):
    size1 = len(in_vals)
    size2 = np.max([len(x) for x in in_vals])
    return pad_2d_vals(in_vals, size1, size2, dtype=dtype)


def pad_2d_vals(in_vals, dim1_size, dim2_size, dtype=np.int32):
    out_val = np.zeros((dim1_size, dim2_size), dtype=dtype)
    if dim1_size > len(in_vals):
        dim1_size = len(in_vals)
    for i in range(dim1_size):
        cur_in_vals = in_vals[i]
        cur_dim2_size = dim2_size
        if cur_dim2_size > len(cur_in_vals):
            cur_dim2_size = len(cur_in_vals)
        out_val[i, :cur_dim2_size] = cur_in_vals[:cur_dim2_size]
    return out_val


def pad_3d_vals_no_size(in_vals, dtype=np.int32):
    size1 = len(in_vals)
    size2 = np.max([len(x) for x in in_vals])
    size3 = 0
    for val in in_vals:
        cur_size3 = np.max([len(x) for x in val])
        if size3 < cur_size3:
            size3 = cur_size3
    return pad_3d_vals(in_vals, size1, size2, size3, dtype=dtype)


def pad_3d_vals(in_vals, dim1_size, dim2_size, dim3_size, dtype=np.int32):
    #     print(in_vals)
    out_val = np.zeros((dim1_size, dim2_size, dim3_size), dtype=dtype)
    if dim1_size > len(in_vals):
        dim1_size = len(in_vals)
    for i in range(dim1_size):
        in_vals_i = in_vals[i]
        cur_dim2_size = dim2_size
        if cur_dim2_size > len(in_vals_i):
            cur_dim2_size = len(in_vals_i)
        for j in range(cur_dim2_size):
            in_vals_ij = in_vals_i[j]
            cur_dim3_size = dim3_size
            if cur_dim3_size > len(in_vals_ij):
                cur_dim3_size = len(in_vals_ij)
            out_val[i, j, :cur_dim3_size] = in_vals_ij[:cur_dim3_size]
    return out_val


def pad_4d_vals(in_vals, dim1_size, dim2_size, dim3_size, dim4_size, dtype=np.int32):
    out_val = np.zeros((dim1_size, dim2_size, dim3_size, dim4_size), dtype=dtype)
    if dim1_size > len(in_vals):
        dim1_size = len(in_vals)
    for i in range(dim1_size):
        in_vals_i = in_vals[i]
        cur_dim2_size = dim2_size
        if cur_dim2_size > len(in_vals_i):
            cur_dim2_size = len(in_vals_i)
        for j in range(cur_dim2_size):
            in_vals_ij = in_vals_i[j]
            cur_dim3_size = dim3_size
            if cur_dim3_size > len(in_vals_ij):
                cur_dim3_size = len(in_vals_ij)
            for k in range(cur_dim3_size):
                in_vals_ijk = in_vals_ij[k]
                cur_dim4_size = dim4_size
                if cur_dim4_size > len(in_vals_ijk):
                    cur_dim4_size = len(in_vals_ijk)
                out_val[i, j, k, :cur_dim4_size] = in_vals_ijk[:cur_dim4_size]
    return out_val


def pad_target_labels(in_val, max_length, dtype=np.float32):
    batch_size = len(in_val)
    out_val = np.zeros((batch_size, max_length), dtype=dtype)
    for i in range(batch_size):
        for index in in_val[i]:
            out_val[i, index] = 1.0
    return out_val
