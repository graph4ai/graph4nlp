from __future__ import print_function
from spodernet.utils.logger import Logger, GlobalLogger
from spodernet.utils.util import save_data, load_data, get_data_path
from os.path import join
from scipy.sparse import csr_matrix

import pytest
import numpy as np
import uuid
import os
import shutil


def test_global_logger():
    log1 = Logger('test1.txt')
    log2 = Logger('test2.txt')
    log1.info('uden')
    log2.info('kek')
    log2.info('rolfen')
    log1.info('keken')


    GlobalLogger.flush()

    expected = ['uden', 'kek', 'rolfen', 'keken']
    with open(GlobalLogger.global_logger_path) as f:
        data = f.readlines()



    print(len(data))
    for i, line in enumerate(data[-4:]):
        message = line.split(':')[3].strip()
        assert message == expected[i]

    assert i  == len(expected) - 1

test_data = [np.float32, np.float64, np.int32, np.int64]
ids = ['float32', 'float64', 'int32', 'int64']
@pytest.mark.parametrize("dtype", test_data, ids=ids)
def test_save_load_data(dtype):
    folder = join(get_data_path(), 'test_hdf')
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.mkdir(folder)
    for i in range(5):
        filename = str(uuid.uuid4())
        data1 = dtype(np.random.randn(100,100))
        save_data(join(folder, filename), data1)
        data2 = load_data(join(folder, filename))
        np.testing.assert_array_equal(data1, data2, 'Arrays must be equal')
    shutil.rmtree(folder)


test_data = [np.float32, np.float64, np.int32, np.int64]
ids = ['float32', 'float64', 'int32', 'int64']
@pytest.mark.parametrize("dtype", test_data, ids=ids)
def test_sparse_save_load_data(dtype):
    folder = join(get_data_path(), 'test_hdf')
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.mkdir(folder)
    for i in range(5):
        filename = str(uuid.uuid4())
        data1 = csr_matrix(dtype(np.random.randn(100,100)))
        save_data(join(folder, filename), data1)
        data2 = load_data(join(folder, filename))
        np.testing.assert_array_equal(data1.toarray(), data2, 'Arrays must be equal')
    shutil.rmtree(folder)

