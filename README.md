# PromptMIA experiments

Standalone code for Figures **1–21**, Tables **1–9**, and the key-generation FLOP
count for the paper [Leveraging Soft Prompts for Privacy Attacks in Federated Prompt Tuning](https://arxiv.org/abs/2601.06641), accepted to NeurIPS 2026.

Each figure and table has a separate script and configuration:

```text
scripts/fig1.py       ... scripts/fig21.py
scripts/table1.py     ... scripts/table9.py
configs/figure1.json  ... configs/figure21.json
configs/table1.json   ... configs/table9.json
```

The [output map](docs/output-map.md) lists every script and its experiment settings.

## Install

Use Python 3.11 and run commands from this directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/check_environment.py
```

GPU experiments require a CUDA-enabled PyTorch installation. The pinned versions
are PyTorch 2.7.1, torchvision 0.22.1, and scikit-learn 1.5.2; the remaining
dependencies are listed in `requirements.txt`.

## Prepare datasets and models

```bash
python scripts/prepare_data.py download --datasets cifar10 cifar100 fashionmnist tinyimagenet
python scripts/download_models.py --models all
```

Follow [docs/data.md](docs/data.md) to import MNIST-M, CINIC-10, MMAFEDB, and the
UPMC Food-101 image/text pairs. Prepare Fashion-Product for the CLIP panel with:

```bash
python scripts/prepare_data.py build-fashion --device cuda
python scripts/prepare_data.py validate --datasets cifar10 cifar100 tinyimagenet fourdataset food101 fashion
```

Data and pretrained weights live in `data/` and `models/`. The vision datasets are
cached as uint8 images in CPU memory by default, with batches transferred to the
selected device. FourDataset's training image cache uses about 18 GB, with
additional memory needed for loading buffers, test data, queries, and the model.

## Run a figure or table

For example, generate Figure 4 and Table 2:

```bash
python scripts/fig4.py --device cuda
python scripts/table2.py --device cuda
```

Each script performs its required training, evaluation, and output generation.
Shared jobs are saved and reused when another figure or table needs them.

To inspect the required jobs before running an experiment:

```bash
python scripts/fig17.py --dry-run
python scripts/table7.py --dry-run
```

To run a figure in separate stages:

```bash
python scripts/fig20.py --stage train --device cuda
python scripts/fig20.py --stage evaluate --device cuda
python scripts/fig20.py --stage plot
```

The same options work for every numbered script. `--device cuda:1` selects another
GPU. `--stage plot` reads the saved measurements and needs no GPU. `--force`
reruns completed jobs in the chosen stage. For example:

```bash
python scripts/fig4.py --stage plot --force
```

Completed jobs are skipped when their configuration, relevant source files, and
required outputs still match. An interrupted training job restarts from its seed;
it does not resume midway through a round. Use `--force` after replacing data or
pretrained weights. Run one writing process per project directory so shared jobs
are not overwritten concurrently.

To adjust an experiment, edit its configuration or pass a separate file:

```bash
python scripts/fig4.py --config configs/my_figure4.json --device cuda
```

`configs/paper.json` is the complete settings catalog. Each numbered script reads
its own configuration file, so changes for an individual run belong in that file.

## Saved outputs

| Directory | Contents |
| --- | --- |
| `runs/<job>/` | Training configuration, accuracy, client partition, global keys, trainable state, and selected checkpoints |
| `cache/queries/` | Frozen queries shared by experiments with the same data and encoder |
| `results/` | Full precision measurements and individual repeated-game results |
| `figures/` | Each figure as PNG and PDF, with its measured summaries in JSON; Figures 2 and 3 have one image per subfigure (`figure2a`, ...) |
| `tables/` | Each table as LaTeX, CSV, and JSON |
| `logs/` | Job completion records and execution journal |

Figures 1 and 8 use the included workflow illustrations. Table 3 contains the
qualitative method comparison. Numerical figures and tables are generated from
the experiment outputs saved by these scripts.

For the arithmetic count in the computational-overhead paragraph:

```bash
python scripts/key_generation_flops.py
```

## Ethical considerations

**Intended use.** PromptMIA is intended solely for privacy auditing and security
evaluation of federated prompt-tuning (FPT) systems. We publish it to expose a new
attack surface and to dispel the assumption that FPT guarantees data protection.
While FPT reduces privacy risk by not sharing raw data, it does not eliminate that
risk. All our experiments use public benchmark datasets in simulated federations;
no deployed system was attacked, and no inference was performed against any real
individual's data.

**Practical safeguards.** Clients should not treat a central aggregator as a
trusted party, and should adopt additional safeguards in sensitive deployments
despite their costs. Mechanisms that prevent the server from observing individual
client updates, such as secure aggregation (Appendix A.6 of the
[paper](https://arxiv.org/abs/2601.06641)) or multi-party management of the prompt
pool, reduce the attack from client-level to group-level membership inference, at
additional communication and computation cost.

**Responsible disclosure.** PromptMIA targets the design of the prompt-selection
protocol in FPT rather than a defect in a particular implementation, and we are not
aware of a commercial FPT deployment with a maintainer to whom conventional
embargoed disclosure would apply. In place of vendor notification, we publish the
attack so that the risk is known to researchers and implementers before such
systems reach wider deployment. We release this code with a clear intended-use
statement, and we document the safeguards above together with their costs and
limits, so that practitioners are not left with a demonstrated attack and no
guidance.

## License

This code is released under the [MIT License](LICENSE).
