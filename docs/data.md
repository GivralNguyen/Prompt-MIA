# Data and pretrained models

Run preparation commands from the project directory. Training reads from `data/`
and `models/`; it does not download datasets. The preparation scripts perform the
explicit downloads. Set `--root` to choose another data location and pass the same
`data_root` in the training configuration.

## Model weights

```bash
python scripts/download_models.py --models all
```

This prepares torchvision ViT-B/32, timm DeiT-B/16 and ConViT-Base, the ViLT
`vilt_200k_mlm_itm.ckpt` checkpoint with its BERT tokenizer, and
`openai/clip-vit-base-patch32`. Choose `vision`, `vilt`, or `clip` to prepare only
one group. All model downloads stay inside `models/`.

## CIFAR and TinyImageNet

```bash
python scripts/prepare_data.py download --datasets cifar10 cifar100 fashionmnist tinyimagenet
python scripts/prepare_data.py validate --datasets cifar10 cifar100 tinyimagenet
```

CIFAR-10 and CIFAR-100 use their official train/test splits. TinyImageNet uses
the official training split and the validation images with `val_annotations.txt`
as the test split. Fashion-MNIST supplies one component of FourDataset.

## FourDataset

Prepare the following layout:

```text
data/
  MNISTM/processed/mnist_m_train.pt
  MNISTM/processed/mnist_m_test.pt
  FashionMNIST/FashionMNIST/raw/...
  cinic10-py/train/<class>/*.png
  cinic10-py/test/<class>/*.png
  MMAFEDB/train/<class>/*.png
  MMAFEDB/test/<class>/*.png
```

MNIST-M tensors contain `(uint8_RGB_images, integer_labels)`. As an alternative,
MNIST-M can use `MNISTM/train/<class>/` and `MNISTM/test/<class>/`. Use the standard
MNIST-M and CINIC-10 splits and the
[MMAFEDB clean dataset](https://www.kaggle.com/datasets/yuulind/mmafedb-clean).
Keep the dataset's class folder names and filename order: the subset selection
takes the first class-quota examples in dataset order.

The loader selects at most 3,000 training and 250 test images per class for each
10-class source. MMAFEDB has seven classes, so its quotas are 4,286 training and
358 test images per class. Classes with fewer images contribute every available
image; no samples are duplicated. With the specified data, the resulting counts
are:

| Source | Training | Test | Label range |
| --- | ---: | ---: | --- |
| MNIST-M | 30,000 | 2,500 | 0–9 |
| Fashion-MNIST | 30,000 | 2,500 | 10–19 |
| CINIC-10 | 30,000 | 2,500 | 20–29 |
| MMAFEDB | 28,629 | 2,506 | 30–36 |
| Total | 118,629 | 10,006 | 37 classes |

```bash
python scripts/prepare_data.py validate --datasets fourdataset
```

The selected source indices and class counts are saved in
`data/splits/fourdataset_train.json` and `fourdataset_test.json`. These records also
help check that a separately prepared copy uses the same rows. Each source is
partitioned among 20 clients in the default 80-client federation.

You can import image data from a CSV or JSONL manifest with `image,label` fields:

```bash
python scripts/prepare_data.py import-images --dataset cinic10 --split train --manifest /path/to/train.csv
```

Choose `mnistm`, `cinic10`, or `mmafedb` and repeat for the test split. Use numeric
labels corresponding to the source's class order and list rows in the intended
within-class filename order. Relative image paths resolve against the manifest
directory, or the optional `--image-root`.

## ViLT: UPMC Food-101

Place `food101_train.arrow` and `food101_test.arrow` in `data/food101/`. Each Arrow
table contains `image` bytes, `text`, and integer `label` columns. The experiment
uses class IDs 0 through 7, retaining the source row order.

Alternatively, import CSV or JSONL records with `image,text,label` fields:

```bash
python scripts/prepare_data.py import-food101 --split train --manifest /path/to/train.csv
python scripts/prepare_data.py import-food101 --split test --manifest /path/to/test.csv
python scripts/prepare_data.py validate --datasets food101
```

The two experiments share the same examples. The text-only setting substitutes
an all-ones image tensor. Both settings use a max-pooled input-embedding query,
frequency-penalized selection during training, eight classes, and 20 clients.

## CLIP: Fashion-Product

```bash
python scripts/download_models.py --models clip
python scripts/prepare_data.py build-fashion --device cuda
python scripts/prepare_data.py validate --datasets fashion
```

The builder reads `ashraq/fashion-product-images-small`, chooses the ten most
frequent `subCategory` labels, and keeps at most 3,000 examples per category. It
uses NumPy seed 0 to select and shuffle examples, then assigns the first 20% to
the test split. Product titles are tokenized to 24 tokens. The frozen CLIP image
and text embeddings are computed once and saved alongside token IDs and labels
in `data/fashionmm_clip/train.npz` and `test.npz`. The preparation manifest records
class names, selected row indices, and split sizes.

The CLIP experiment uses both image and title inputs with 80 clients. Its query
is the sum of the individually normalized image and text embeddings. Dataset
downloads are cached inside `data/downloads/`.
