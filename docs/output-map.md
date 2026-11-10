# Paper output map

Each numbered output has its own script and configuration. For example,
`python scripts/fig17.py --dry-run` lists the jobs for Figure 17, and
`python scripts/table2.py` trains and evaluates the models required for Table 2.
Use `--stage train`, `--stage evaluate`, or `--stage plot` to run one stage.
Finished shared jobs are reused when another output requires the same settings.

The scope is Figures 1–21 and Tables 1–9 of the paper. The configurations
contain all training and evaluation settings needed by each output.

## Figures

| Figure | Script | Contents and settings |
|---|---|---|
| 1 | `scripts/fig1.py` | Diagram of the PromptMIA attack workflow. |
| 2 | `scripts/fig2.py` | CIFAR-10, ViT-B/32, round 60: benign keys, naive injection, and PromptMIA injection; training target 0. Writes `figure2a`, `figure2b`, and `figure2c`. Needs a GPU to encode the target query. |
| 3 | `scripts/fig3.py` | CIFAR-10, ViT-B/32, round 60: keys with the first 200 training queries, five K-means clusters, and the Gaussian mixture visualization. Writes `figure3a`, `figure3b`, and `figure3c`. Needs a GPU to encode the queries. |
| 4 | `scripts/fig4.py` | PromptMIA and naive attack Advantage/ASR on four datasets, averaged over ViT-B/32, ConViT-Base, and DeiT-B/16. |
| 5 | `scripts/fig5.py` | Advantage of PromptMIA, Blackbox-Loss, Loss-Series, and FedMIA-I at batch size 64: matched-FPR and best-threshold panels, averaged over three backbones. Members and batch fillers come from target client 0. |
| 6 | `scripts/fig6.py` | ViT-B/32, mean over four datasets: no defense, 80% gradient sparsification, Mixup with alpha 0.2, and adaptive data augmentation. |
| 7 | `scripts/fig7.py` | Adaptive input-noise attack on ViT-B/32: epsilon 3, 5, and 8 plus the clean reference, shown separately for each dataset. |
| 8 | `scripts/fig8.py` | Diagram of federated prompt tuning. |
| 9 | `scripts/fig9.py` | ViT-B/32, mean over four datasets: dropout rates 0.2, 0.3, and 0.4. |
| 10 | `scripts/fig10.py` | DeiT-B/16 per-sample FPR bound, averaged over 100 test targets: four datasets at alpha 0.5; CIFAR-100 at alpha 0.05, 0.1, and 0.5. |
| 11 | `scripts/fig11.py` | CIFAR-100 with extreme non-IID clients and ViT-B/32: Gaussian mixture panels for pool sizes 10, 20, 30, and 40. |
| 12 | `scripts/fig12.py` | PromptMIA and naive attack Advantage/ASR for ViT-B/32 on all four datasets. |
| 13 | `scripts/fig13.py` | PromptMIA and naive attack Advantage/ASR for ConViT-Base on all four datasets. |
| 14 | `scripts/fig14.py` | PromptMIA and naive attack Advantage/ASR for DeiT-B/16 on all four datasets. |
| 15 | `scripts/fig15.py` | Four outlier detectors on the CIFAR-10 ViT-B/32 pool with PromptMIA injection for training target 0. |
| 16 | `scripts/fig16.py` | The same detectors on the benign CIFAR-10 ViT-B/32 pool. |
| 17 | `scripts/fig17.py` | Six ViT-B/32 ablations, each averaged over four datasets: pool size, selection size, minimum margin, interval width, training round, and local epochs. |
| 18 | `scripts/fig18.py` | CIFAR-10 ViT-B/32 key/query distributions at rounds 0, 1, 20, and 60, of the default model (key learning rate 1e-3). |
| 19 | `scripts/fig19.py` | PromptMIA Advantage/ASR up to batch size 1024, averaged over all three backbones, with one panel per dataset. |
| 20 | `scripts/fig20.py` | ViLT on eight UPMC Food-101 classes with image/text and text-only inputs; CLIP ViT-B/32 on Fashion-Product image/title pairs. |
| 21 | `scripts/fig21.py` | ViT-B/32 ASR with 80, 160, and 320 clients, averaged over all four datasets. |

## Tables

| Table | Script | Contents and settings |
|---|---|---|
| 1 | `scripts/table1.py` | Outlier precision, recall, and F1 averaged over three datasets and three backbones, with injection rate 0.1 and 100 training targets per model. Parenthesized values are population SD across the nine model/dataset combinations. |
| 2 | `scripts/table2.py` | ViT-B/32 clean-test accuracy under input-noise training, epsilon 3, 5, and 8 and no noise, on four datasets. Mean and sample SD over training seeds 0, 1, and 2. |
| 3 | `scripts/table3.py` | Qualitative comparison of the attack surfaces, methodologies, and assumptions of the MIA methods. |
| 4 | `scripts/table4.py` | Norm-rescaled injection and cross-round key-drift auditing: F1 at injection rates 1/10 and 1/60 and thresholds median + c times MAD, c in 1, 2, 3. Mean over four datasets. |
| 5 | `scripts/table5.py` | Extreme non-IID ASR on CIFAR-100 and TinyImageNet, ViT-B/32, pool sizes 10, 20, 30, and 40, batches 64 and 256. |
| 6 | `scripts/table6.py` | Per-model outlier precision, recall, and F1 for CIFAR-10, CIFAR-100, and TinyImageNet with three backbones and four detectors; injection rate 0.1. |
| 7 | `scripts/table7.py` | Target-distinctiveness quintiles ranked by nearest-key cosine similarity: FPR/ASR at batches 64 and 256, averaged over four datasets. Each dataset uses 2,000 test targets and 32 client batches per target. |
| 8 | `scripts/table8.py` | Final accuracy at round 60, with and without injection at round 30: CIFAR-10/DeiT-B/16 and FourDataset/ViT-B/32, seed 0. |
| 9 | `scripts/table9.py` | Parameter counts and attack sparsity for the three vision backbones, pool size 20, four modified keys, and a ten-class classifier. |

`python scripts/key_generation_flops.py` computes the arithmetic count reported
in the computational-overhead paragraph for four 768-dimensional adversarial keys
and sixteen benign keys.

