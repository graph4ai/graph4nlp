'''
Created on Nov, 2018

@author: hugo

'''
import json
import numpy as np


def dump_ndarray(data, path_to_file):
    try:
        with open(path_to_file, 'wb') as f:
            np.save(f, data)
    except Exception as e:
        raise e

def load_ndarray(path_to_file):
    try:
        with open(path_to_file, 'rb') as f:
            data = np.load(f)
    except Exception as e:
        raise e

    return data

def dump_ndjson(data, file):
    try:
        with open(file, 'w') as f:
            for each in data:
                f.write(json.dumps(each) + '\n')
    except Exception as e:
        raise e

def load_ndjson(file, return_type='array'):
    if return_type == 'array':
        return load_ndjson_to_array(file)
    elif return_type == 'dict':
        return load_ndjson_to_dict(file)
    else:
        raise RuntimeError('Unknown return_type: %s' % return_type)

def dump_json(data, file, indent=None):
    try:
        with open(file, 'w') as f:
            json.dump(data, f, indent=indent)
    except Exception as e:
        raise e

def load_json(file):
    try:
        with open(file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        raise e
    return data
