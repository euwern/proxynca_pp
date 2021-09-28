
ProxyNCA++: Revisiting and Revitalizing Proxy Neighborhood Component Analysis
==============================================================================
This repo consists of the source code for the [ProxyNCA++ paper](https://arxiv.org/abs/2004.01113)

<img src="https://i.imgur.com/gCc3EmZ.png" title='Components of ProxyNCA++]' width="600">

Make sure to download the corresponding dataset to the correct folder as specified in dataset/config.json
We also include script to convert the dataset to hdf5 format.

To create data
```
> conda activate pytorch_p36

# create Shoes_train & Shoes_test files. This command outputs TRAIN_DATA_SIZE & TEST_DATA_SIZE
> python data_to_txt.py --path DATA_DIR

# create hdf5 file
> python dataset/make_shoes_hdf5.py --nb_train_all TRAIN_DATA_SIZE --nb_test_all TEST_DATA_SIZE --source DATA_DIR --output OUTPUT_PATH
```

To start training

nb_train_all: 106974
nb_test_all: 106550

```

> conda activate pytorch_p36
> 

```

The following is the Bibtex of our paper:
```
@article{teh2020proxynca++,
  title={ProxyNCA++: Revisiting and Revitalizing Proxy Neighborhood Component Analysis},
  author={Teh, Eu Wern and DeVries, Terrance and Taylor, Graham W},
  journal={arXiv preprint arXiv:2004.01113},
  year={2020}
}
```
# Önce train, sonrasında ise trainval çalıştıralacak.
python dataset/make_mixed_hdf5.py --nb_train_all 22888 --nb_test_all 22478 --source /home/counterfake/workstation/datasets/proxy/mixed_v0_1 --output /home/counterfake/workstation/datasets/proxy/mixed_v0_1 