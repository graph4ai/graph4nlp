import platform
import os

from setuptools import setup, find_packages
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

cuda_versions = {
    '1': '9.2',
    '2': '10.1',
    '3': '10.2',
    '4': '11.0',
    '5': '11.1',
    '6': 'cpu'
}


class MyBdistWheel(_bdist_wheel):
    def finalize_options(self):
        _bdist_wheel.finalize_options(self)
        self.root_is_pure = False


if __name__ == '__main__':
    os_tag = {
        'Windows': 'win_amd64',
        'Darwin': 'macosx_x86_64',
        'Linux': 'manylinux1_x86_64'
    }

    # Parse config
    try:
        with open('config', 'r') as f:
            version = f.readlines()[0].strip()
    except:
        raise FileNotFoundError('config file not found. Please run ./configure first.')

    cuda_version = cuda_versions[version]

    if cuda_version == '11.1':
        os.system("pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html")

    if cuda_version == 'none':
        cuda_version = ''
    elif cuda_version.find('.') != -1:
        cuda_version = '-cu' + ''.join(cuda_version.split('.'))
    else:
        cuda_version = ''

    install_requirement = ['pythonds', 'nltk >= 3.5', 'stanfordcorenlp', 'scipy >= 1.5.2',
                           'scikit-learn >= 0.23.2', 'networkx >= 2.5', 'dgl{} >= 0.4'.format(cuda_version),
                           'ogb', 'torchtext', 'tqdm >= 4.29.0', 'pyyaml']
    pytorch_requirement = 'torch >= 1.6.0' if platform.system() != 'Windows' else 'torch >= 1.8.1'
    print("System: {}. PyTorch Requirement = {}".format(platform.system(), pytorch_requirement))

    install_requirement.append(pytorch_requirement)
    if cuda_version in ['-cu101', '-cu92']:
        install_requirement.append('torch <= 1.7.0')

    setup(
        name='graph4nlp{}'.format(cuda_version),
        version='0.2a5',
        description='A DGL and PyTorch based graph deep learning library for natural language processing',
        author='Graph4NLP Team',
        license='MIT',
        include_package_data=True,
        packages=find_packages('.', exclude=(
            "examples.*", "examples", "graph4nlp.pytorch.test.*", "graph4nlp.pytorch.test")),
        install_requires=install_requirement,
        platforms=os_tag[platform.system()]
    )
    print("Graph4NLP Python library installation finished. Please manually check Stanford CoreNLP"
          "(https://stanfordnlp.github.io/CoreNLP/) is installed and "
          "running in your environment.")
