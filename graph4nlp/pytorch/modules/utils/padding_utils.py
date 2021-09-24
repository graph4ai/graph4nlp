import numpy as np
import torch

from typing import List


def pad_4d_vals_sparse(in_vals: List[List[List[np.ndarray]]],
                       dim1: int, dim2: int, dim3: int, dim4: int) -> torch.Tensor:
    """
    Receives a list of lists of lists of ndarray, produces a sparse tensor of shape dim1, dim2, dim3, dim4
    in_vals[0][0][0] is a ndarray of shape (NB_ELTS,).
    Same as pad_4d_vals, but SPARSE. In the case of BertEmbeddings, the arrays are mostly made of zeros...
    """
    # Step 1 - build the indices of the non-zero elements
    indices = [[], [], [], []]
    nb_values = 0
    for i in range(len(in_vals)):
        for j in range(len(in_vals[i])):
            for k in range(len(in_vals[i][j])):
                array_ijk = in_vals[i][j][k]
                nz = array_ijk.nonzero()
                for m in nz[0]:
                    indices[0].append(i)
                    indices[1].append(j)
                    indices[2].append(k)
                    indices[3].append(m)
                    nb_values += 1
    indices_t = torch.LongTensor(indices)

    # The values are only ONES
    values = [1.0] * nb_values

    out = torch.sparse_coo_tensor(
        indices=indices_t,
        values=values,
        size=(dim1, dim2, dim3, dim4),
        dtype=torch.float32
    )
    return out


def make_batches(size, batch_size):
    nb_batch = int(np.ceil(size/float(batch_size)))
    return [(i*batch_size, min(size, (i+1)*batch_size)) for i in range(0, nb_batch)] # zgwang: starting point of each batch

def pad_2d_vals_no_size(in_vals, dtype=np.int32):
    size1 = len(in_vals)
    size2 = np.max([len(x) for x in in_vals])
    return pad_2d_vals(in_vals, size1, size2, dtype=dtype)

def pad_2d_vals(in_vals, dim1_size, dim2_size, dtype=np.int32):
    out_val = np.zeros((dim1_size, dim2_size), dtype=dtype)
    if dim1_size > len(in_vals): dim1_size = len(in_vals)
    for i in range(dim1_size):
        cur_in_vals = in_vals[i]
        cur_dim2_size = dim2_size
        if cur_dim2_size > len(cur_in_vals): cur_dim2_size = len(cur_in_vals)
        out_val[i,:cur_dim2_size] = cur_in_vals[:cur_dim2_size]
    return out_val

def pad_3d_vals_no_size(in_vals, dtype=np.int32):
    size1 = len(in_vals)
    size2 = np.max([len(x) for x in in_vals])
    size3 = 0
    for val in in_vals:
        cur_size3 = np.max([len(x) for x in val])
        if size3<cur_size3: size3 = cur_size3
    return pad_3d_vals(in_vals, size1, size2, size3, dtype=dtype)

def pad_3d_vals(in_vals, dim1_size, dim2_size, dim3_size, dtype=np.int32):
#     print(in_vals)
    out_val = np.zeros((dim1_size, dim2_size, dim3_size), dtype=dtype)
    if dim1_size > len(in_vals): dim1_size = len(in_vals)
    for i in range(dim1_size):
        in_vals_i = in_vals[i]
        cur_dim2_size = dim2_size
        if cur_dim2_size > len(in_vals_i): cur_dim2_size = len(in_vals_i)
        for j in range(cur_dim2_size):
            in_vals_ij = in_vals_i[j]
            cur_dim3_size = dim3_size
            if cur_dim3_size > len(in_vals_ij): cur_dim3_size = len(in_vals_ij)
            out_val[i, j, :cur_dim3_size] = in_vals_ij[:cur_dim3_size]
    return out_val

def pad_4d_vals(in_vals, dim1_size, dim2_size, dim3_size, dim4_size, dtype=np.int32):
    out_val = np.zeros((dim1_size, dim2_size, dim3_size, dim4_size), dtype=dtype)
    if dim1_size > len(in_vals): dim1_size = len(in_vals)
    for i in range(dim1_size):
        in_vals_i = in_vals[i]
        cur_dim2_size = dim2_size
        if cur_dim2_size > len(in_vals_i): cur_dim2_size = len(in_vals_i)
        for j in range(cur_dim2_size):
            in_vals_ij = in_vals_i[j]
            cur_dim3_size = dim3_size
            if cur_dim3_size > len(in_vals_ij): cur_dim3_size = len(in_vals_ij)
            for k in range(cur_dim3_size):
                in_vals_ijk = in_vals_ij[k]
                cur_dim4_size = dim4_size
                if cur_dim4_size > len(in_vals_ijk): cur_dim4_size = len(in_vals_ijk)
                out_val[i, j, k, :cur_dim4_size] = in_vals_ijk[:cur_dim4_size]
    return out_val

def pad_target_labels(in_val, max_length, dtype=np.float32):
    batch_size = len(in_val)
    out_val = np.zeros((batch_size, max_length), dtype=dtype)
    for i in range(batch_size):
        for index in in_val[i]:
            out_val[i,index] = 1.0
    return out_val
