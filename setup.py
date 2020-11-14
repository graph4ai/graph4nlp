from setuptools import setup, find_packages

install_requirement = ['torch >= 1.6.0', 'pythonds', 'nltk >= 3.5', 'stanfordcorenlp', 'scipy >= 1.5.2',
                       'scikit-learn >= 0.23.2', 'networkx >= 2.5', 'dgl']

def get_cuda_version():
    import os
    nvsmi = os.popen('nvidia-smi')
    gpu_info_str = nvsmi.read()
    if gpu_info_str == '':
        install_requirement.append('dgl >= 0.4')
    else:
        cuda_version_prompt = 'CUDA Version: '
        version_number_start = gpu_info_str.find(cuda_version_prompt) + len(cuda_version_prompt)
        version_number = gpu_info_str[version_number_start:].split(' ')[0]
        install_requirement.append('dgl-cuda{} >= 0.4'.format(version_number))

packages = find_packages('.')

setup(
    name='graph4nlp',
    version='0.1.3',
    description='GNN for NLP library',
    author='graph4nlp team',
    license='MIT',
    packages=packages,
    install_requires=install_requirement,
)
