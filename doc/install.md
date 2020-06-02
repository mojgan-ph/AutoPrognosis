# Installation instructions

The required dependencies work with python3.6 or python3.7.
Some algorithms require tensorflow, ideally with GPU acceleration.

## Anaconda (Windows, Linux, ...)

Depends upon the Anaconda Distribution (https://www.anaconda.com)

```
   # if you do not have python3.6 or python 3.7 create an environment with the required version of python

conda create -n python36env python=3.6  # install and create an python3.6 environment
conda activate python36env              # use source for linux like environments, for example 'source ~/anaconda3/bin/activate python36env'

   # install packages:
   
conda install pandas scipy pivottablejs matplotlib tqdm requests jupyter seaborn
pip install sklearn gpyopt xgboost sets

```


- install tensorflow (optional)

```
conda install tensorflowXXX keras
```

## Linux

It is known to work and tested with Ubuntu vs 18.04, R vs 3.5.1,
python3.6 and tensorflow (1.4.1 and higher)


- install the python dependencies with the following steps:

```
pip3 install -r cfg/requirements.txt
```

- install tensorflow

choose your version of tensorflow (CPU/GPU) and install with:

```

pip3 install tensorflowXXX keras

```

