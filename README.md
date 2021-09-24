# Explicit Cross-Modal Reasoning for Visual Commonsense Reasoning

The Code for IEEE TMM paper "Explicit Cross-Modal Reasoning for Visual Commonsense Reasoning"

## Setting up and using the repo

1. Get the dataset. Follow the steps in `data/README.md`. This includes the steps to get the pretrained BERT embeddings and the parsed results of sentences.

2. Install cuda 10.0 if it's not available already.

3. Install anaconda if it's not available already, and create a new environment. You need to install a few things, namely, pytorch 1.1.0, torchvision, and allennlp.

```
wget https://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh
conda update -n base -c defaults conda
conda create --name CMR python=3.6
source activate CMR

conda install numpy pyyaml setuptools cmake cffi tqdm pyyaml scipy ipython mkl mkl-include cython typing h5py pandas nltk spacy numpydoc scikit-learn jpeg

conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch

pip install -r allennlp-requirements.txt
pip install --no-deps allennlp==0.8.0
python -m spacy download en_core_web_sm


# this one is optional but it should help make things faster
pip uninstall pillow && CC="cc -mavx2" pip install -U --force-reinstall pillow-simd
```

4. That's it! Now to set up the environment, run `source activate CMR`.

## Train/Evaluate models
Please refer to `models/README.md`.

## Citations
If you make use of this repository for your research, please cite the following paper:

```
@article{zhang2021explicit,
  title={Explicit Cross-Modal Representation Learning for Visual Commonsense Reasoning},
  author={Zhang, Xi and Zhang, Feifei and Xu, Changsheng},
  journal={IEEE Transactions on Multimedia},
  year={2021},
  publisher={IEEE}
}
```

## Acknowledgement
- We refer to the repo [r2c](https://github.com/rowanz/r2c/) and [tab-vcr](https://github.com/Deanplayerljx/tab-vcr) for preprocessing codes.



