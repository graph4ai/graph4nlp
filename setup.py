import platform
from setuptools import find_packages, setup
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

cuda_versions = {"1": "9.2", "2": "10.1", "3": "10.2", "4": "11.0", "5": "11.1", "6": "cpu"}


class MyBdistWheel(_bdist_wheel):
    def finalize_options(self):
        _bdist_wheel.finalize_options(self)
        self.root_is_pure = False


if __name__ == "__main__":
    os_tag = {"Windows": "win_amd64", "Darwin": "macosx_x86_64", "Linux": "manylinux1_x86_64"}

    # Parse config
    try:
        with open("config", "r") as f:
            version = f.readlines()[0].strip()
    except:
        raise FileNotFoundError("config file not found. Please run ./configure first.")

    cuda_version = cuda_versions[version]

    # if cuda_version == '11.1':
    #     os.system("pip install torch==1.8.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html")

    # if platform.system() == 'Windows':
    #     os.system('pip install torch==1.8.1+cu102 -f https://download.pytorch.org/whl/torch_stable.html')

    if cuda_version == "none":
        cuda_version = ""
    elif cuda_version.find(".") != -1:
        cuda_version = "-cu" + "".join(cuda_version.split("."))
    else:
        cuda_version = ""

    install_requirement = [
        "pythonds",
        "nltk >= 3.5",
        "stanfordcorenlp",
        "scipy >= 1.5.2",
        "scikit-learn >= 0.23.2",
        "networkx >= 2.5",
        "dgl{} >= 0.4".format(cuda_version),
        "ogb",
        "tqdm >= 4.29.0",
        "pyyaml",
        "transformers",
    ]
    # pytorch_requirement = 'torch >= 1.6.0' if platform.system() != 'Windows' else 'torch == 1.8.0'
    # torchtext_requirement = 'torchtext >= 0.7.0' if platform.system() != 'Windows' else 'torchtext == 0.9.0'
    # install_requirement.append(pytorch_requirement)
    # install_requirement.append(torchtext_requirement)

    setup(
        name="graph4nlp{}".format(cuda_version),
        version="0.4.0",
        description="A DGL and PyTorch based graph deep learning library for natural language processing",
        author="Graph4NLP Team",
        license="Apache 2.0",
        include_package_data=True,
        packages=find_packages(
            ".",
            exclude=(
                "examples.*",
                "examples",
                "graph4nlp.pytorch.test.*",
                "graph4nlp.pytorch.test",
            ),
        ),
        install_requires=install_requirement,
        platforms=os_tag[platform.system()],
    )
    print(
        "Graph4NLP Python library installation finished. Please manually check Stanford CoreNLP"
        "(https://stanfordnlp.github.io/CoreNLP/) is installed and "
        "running in your environment."
    )
