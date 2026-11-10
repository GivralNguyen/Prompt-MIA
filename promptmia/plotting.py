"""Create the numbered paper figures and tables from measured result files."""

from pathlib import Path
import csv
import json
import math
import shutil

import numpy as np


DATASETS = ('cifar10', 'cifar100', 'tinyimagenet', 'fourdataset')
DATASET_LABELS = dict(cifar10='CIFAR-10', cifar100='CIFAR-100', tinyimagenet='TinyImageNet', fourdataset='FourDataset')
METHOD_LABELS = dict(promptmia='PromptMIA', naive='Naive', blackbox_loss='Blackbox-Loss',
                     loss_series='Loss-Series', fedmia_i='FedMIA-I')
METRIC_LABELS = dict(asr='Attack Success Rate', advantage='Advantage', tpr='TPR', fpr='FPR')
# The colours of Figure 3c, then more for up to 22 clusters.
CLUSTER_COLORS = ['#ffb347', '#c9e4ff', '#fff3c4', '#d6d6d6', '#e6d6ff', '#f7c6d9', '#b5e7c4', '#a8dadc', '#f4d35e',
                  '#c9a0dc', '#8e2c7a', '#e07a5f', '#3d8b6e', '#5b8bd6', '#d4a373', '#9d4edd', '#2a9d8f', '#e76f51',
                  '#8ab17d', '#bc6c25', '#6d597a', '#ef476f']


def _read(source, root):
    if source.startswith('train:'):
        name = source.split(':', 1)[1]
        directory = root / 'runs' / name
        config = json.loads((directory / 'config.json').read_text())
        metrics = json.loads((directory / 'metrics.json').read_text())
        return dict(name=source, run=name, kind='training', config=config, metrics=metrics,
                    dataset=config.get('dataset'), backbone=config.get('backbone'))
    result = json.loads((root / 'results' / (source + '.json')).read_text())
    result.setdefault('name', source)
    return result


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def _label(record, field='label'):
    if field == 'label':
        return str(record.get('settings', {}).get('label', record.get('label', record['name'])))
    if field == 'round':
        return f"Round {record.get('round', 60)}"
    value = record.get(field, record.get('config', {}).get(field, record.get('settings', {}).get(field)))
    return str(value)


def summarize_attack(records, method, metric):
    """Average runs equally; a single run retains its repeated-game error band."""
    if not records:
        raise ValueError('A curve needs at least one result file')
    batch_sizes = records[0]['batch_sizes']
    means, within = [], []
    for record in records:
        if record['batch_sizes'] != batch_sizes:
            raise ValueError('Every result averaged into a curve must use the same batch sizes')
        raw = record['methods'][method]
        values = [np.asarray(raw[str(b)][metric], dtype=float) for b in batch_sizes]
        means.append([float(x.mean()) for x in values])
        within.append([float(x.std(ddof=0)) for x in values])
    values = np.asarray(means)
    return dict(batch_sizes=batch_sizes, mean=values.mean(0).tolist(),
                std=(values.std(0) if len(values) > 1 else np.asarray(within[0])).tolist(),
                source_means=means, sources=[r['name'] for r in records])


def _curve_definitions(panel, records):
    if panel.get('curves'):
        return panel['curves']
    groups = {}
    for r in records:
        key = _label(r, panel['group_by']) if panel.get('group_by') else ''
        groups.setdefault(key, []).append(r['name'])
    definitions = []
    methods = panel.get('methods', ['promptmia'])
    metrics = panel.get('metrics', ['asr'])
    for group, sources in groups.items():
        for method in methods:
            for metric in metrics:
                label = group
                if len(methods) > 1 or not group:
                    label = ' - '.join(x for x in [label, METHOD_LABELS.get(method, method)] if x)
                if len(metrics) > 1:
                    label += ' - ' + METRIC_LABELS.get(metric, metric)
                definitions.append(dict(label=label, sources=sources, method=method, metric=metric))
    return definitions


def _default_panels(spec, records):
    figure = int(spec['id'].removeprefix('figure'))
    if figure in (4, 6, 7, 12, 13, 14):
        panels = []
        for dataset in DATASETS:
            sources = [r['name'] for r in records if r.get('dataset') == dataset]
            if not sources:
                continue
            panels.append(dict(title=DATASET_LABELS[dataset], sources=sources,
                               methods=['promptmia', 'naive'] if figure in (4, 12, 13, 14) else ['promptmia'],
                               metrics=['advantage', 'asr'] if figure in (4, 12, 13, 14) else ['asr'],
                               group_by='label' if figure in (6, 7) else None))
        return panels
    return [dict(title=spec.get('title', ''), sources=[r['name'] for r in records],
                 group_by='label', metrics=['asr'])]


def _line_figure(spec, records, plt):
    panels = spec.get('panels') or _default_panels(spec, records)
    ncols = min(int(spec.get('ncols', 3 if spec['id'] == 'figure17' else len(panels))), len(panels))
    nrows = math.ceil(len(panels) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=spec.get('figsize', (5.5*ncols, 4*nrows)), squeeze=False)
    lookup = {r['name']: r for r in records}
    output = []
    for ax, panel in zip(axes.flat, panels):
        chosen = [lookup[name] for name in panel.get('sources', lookup)]
        curves = _curve_definitions(panel, chosen)
        result = dict(title=panel.get('title', ''), curves=[])
        for i, curve in enumerate(curves):
            selected = [lookup[name] for name in curve['sources']]
            method, metric = curve.get('method', 'promptmia'), curve.get('metric', 'asr')
            summary = summarize_attack(selected, method, metric)
            sizes = summary['batch_sizes']
            x = np.arange(len(sizes)) if panel.get('categorical_x', spec.get('categorical_x', False)) else sizes
            if panel.get('categorical_x', spec.get('categorical_x', False)):
                ax.set_xticks(x, [str(b) for b in sizes])
            mean, std = np.asarray(summary['mean'])*100, np.asarray(summary['std'])*100
            if not panel.get('group_by') and not panel.get('curves'):
                color = 'tab:blue' if method == 'promptmia' else 'tab:red'
                ls, marker = ('-', 'o') if metric == 'advantage' else ('--', '^')
            else:
                color, ls, marker = f'C{i % 10}', '-', ('o', 's', '^', 'D', 'v')[i % 5]
            ax.plot(x, mean, label=curve['label'], color=color, linestyle=ls, marker=marker, linewidth=1.8)
            if panel.get('bands', spec.get('bands', spec['id'] != 'figure4')):
                ax.fill_between(x, mean-std, mean+std, color=color, alpha=.16)
            result['curves'].append(dict(curve, **summary))
        ax.set_title(panel.get('title', ''))
        ax.set_xlabel('Batch Size')
        ax.set_ylabel('Percentage (%)')
        limits = panel.get('ylim', spec.get('ylim', [0, 102]))
        if limits is not None:
            ax.set_ylim(*limits)
        ax.grid(alpha=.35)
        ax.legend(fontsize=8)
        output.append(result)
    for ax in list(axes.flat)[len(panels):]:
        ax.set_visible(False)
    if spec.get('title'):
        fig.suptitle(spec['title'])
    fig.tight_layout()
    return fig, dict(panels=output)


def _baseline_figure(spec, records, plt):
    methods = ('promptmia', 'blackbox_loss', 'loss_series', 'fedmia_i')
    datasets = [d for d in DATASETS if any(r.get('dataset') == d for r in records)]
    batch = int(spec.get('batch_size', 64))
    colors = dict(promptmia='#1f77b4', blackbox_loss='#9ecae1', loss_series='#ffbb78', fedmia_i='#ff7f0e')
    labels = dict(DATASET_LABELS, cifar10='CIFAR10', cifar100='CIFAR100')
    font = 18
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    x, width = np.arange(len(datasets)), .2
    summary = {}
    for ax, operating_point, title in zip(axes, ('matched', 'best'),
                                         ("Advantage at PromptMIA's FPR", 'Advantage at Best Threshold (Upper Bound)')):
        data = {}
        for i, method in enumerate(methods):
            data[method] = {}
            for dataset in datasets:
                # A loss result may carry the same PromptMIA measurement as its source job.
                candidates = {r.get('run', r['name']): r for r in records
                              if r.get('dataset') == dataset and method in r.get('methods', {})}
                values = []
                for record in candidates.values():
                    measured = record['methods'][method]
                    if method == 'promptmia' and 'advantage' not in measured:
                        measured = {m: float(np.mean(v)) for m, v in measured[str(batch)].items()}
                    elif method != 'promptmia' and operating_point == 'matched':
                        measured = measured['at_fpr']
                    values.append(measured['advantage'])
                if not values:
                    raise ValueError(f'Figure 5 is missing {method} results for {dataset}')
                data[method][dataset] = dict(mean=float(np.mean(values)), std=float(np.std(values)))
            ax.bar(x+(i-1.5)*width, [data[method][d]['mean'] for d in datasets], width,
                   yerr=[data[method][d]['std'] for d in datasets], capsize=3, error_kw=dict(lw=1),
                   label=METHOD_LABELS[method], color=colors[method])
        ax.set_title(title, fontweight='bold', fontsize=font)
        ax.set_ylabel('Advantage Score', fontsize=font)
        ax.set_ylim(0, 1.1)
        ax.set_xticks(x)
        ax.set_xticklabels([labels[d] for d in datasets], rotation=15, ha='right', fontsize=font)
        ax.tick_params(axis='y', labelsize=font)
        ax.grid(axis='y', ls='--', alpha=.4)
        summary[operating_point] = data
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc='upper center', ncol=4, bbox_to_anchor=(.5, 1.05), fontsize=font)
    fig.tight_layout(rect=(0, 0, 1, .93))
    return fig, dict(batch_size=batch, operating_points=summary)


def _bound_panels(spec, records, root, plt):
    """One image per subfigure (figure10a, figure10b): the mean bound over the targets, one bar per setting, log axis."""
    lookup = {r['name']: r for r in records}
    folder = root / 'figures'
    folder.mkdir(parents=True, exist_ok=True)
    paths, output = [], []
    for letter, panel in zip('abcdefgh', spec['panels']):
        values = [lookup[n]['log10_mean_bound'] for n in panel['sources']]
        fig, ax = plt.subplots(figsize=(6, 4.5))
        lower = math.floor(min(values))-3
        upper = math.ceil(max(values))
        x = np.arange(len(values))
        ax.bar(x, [v-lower for v in values], bottom=lower, color=panel['colors'][:len(values)], edgecolor='black', width=.6)
        for i, value in enumerate(values):
            exponent = math.floor(value)
            ax.text(i, value+.3, f'{10**(value-exponent):.2f}e{exponent}', ha='center', va='bottom', fontsize=9,
                    fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(panel['labels'])
        ax.set_ylim(lower, upper+3)
        ticks = np.arange(lower, upper+4, 2 if upper+3-lower <= 14 else 4)
        ax.set_yticks(ticks)
        ax.set_yticklabels([f'$10^{{{int(v)}}}$' for v in ticks])
        ax.set_ylabel('FPR Bound')
        ax.set_title(panel['title'], fontsize=10)
        if panel.get('xlabel'):
            ax.set_xlabel(panel['xlabel'])
        ax.grid(True, axis='y', alpha=.3)
        paths += _save_panel(fig, folder / (spec['id']+letter))
        plt.close(fig)
        output.append(dict(title=panel['title'], sources=panel['sources'], labels=panel['labels'],
                           log10_mean_bound=values))
    summary = folder / (spec['id']+'.json')
    _write_json(summary, dict(id=spec['id'], sources=spec.get('sources', []), panels=output))
    return paths+[summary]


COMPARISON_STYLE = {
    ('promptmia', 'advantage'): dict(color='blue', ls='-', marker='o', label='PromptMIA - Advantage'),
    ('promptmia', 'asr'): dict(color='blue', ls='--', marker='^', label='PromptMIA - Attack Success Rate'),
    ('naive', 'advantage'): dict(color='red', ls='-', marker='s', label='Naive - Advantage'),
    ('naive', 'asr'): dict(color='red', ls='--', marker='d', label='Naive - Attack Success Rate')}


def _comparison_row(spec, records, root, plt):
    """One row of per-dataset panels for one backbone: Advantage and ASR of PromptMIA and naive injection against batch
    size, with a band of one std over the game repeats of the single training run."""
    lookup = {r['name']: r for r in records}
    fig, axes = plt.subplots(1, len(spec['panels']), figsize=(20, 4.2))
    output = []
    for ax, panel in zip(axes, spec['panels']):
        record = lookup[panel['sources'][0]]
        curves = []
        for method in panel['methods']:
            batch_sizes = record['batch_sizes']
            for metric in panel['metrics']:
                values = np.array([record['methods'][method][str(b)][metric] for b in batch_sizes])
                mean, std = values.mean(1), values.std(1)
                style = COMPARISON_STYLE[(method, metric)]
                ax.plot(batch_sizes, 100*mean, **style)
                ax.fill_between(batch_sizes, 100*(mean-std), 100*(mean+std), color=style['color'], alpha=0.2, lw=0)
                curves.append(dict(method=method, metric=metric, batch_sizes=batch_sizes, mean=mean.tolist(),
                                   std=std.tolist()))
        ax.set_title(panel['title'])
        ax.set_xlabel('Batch Size')
        ax.set_ylabel('Percentage (%)')
        ax.grid(True)
        output.append(dict(title=panel['title'], sources=panel['sources'], curves=curves))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(spec['title'])
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    stem = root / 'figures' / spec['id']
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix('.pdf'))
    fig.savefig(stem.with_suffix('.png'), dpi=150)
    plt.close(fig)
    _write_json(stem.with_suffix('.json'), dict(id=spec['id'], sources=spec.get('sources', []), panels=output))
    return [stem.with_suffix('.png'), stem.with_suffix('.pdf'), stem.with_suffix('.json')]


def _dataset_mean_panels(spec, records, root, plt):
    """One image per subfigure (figure17a, ...; a single panel is figure21): ASR against batch size, each curve the
    mean over the datasets of each dataset's mean over the game repeats."""
    lookup = {r['name']: r for r in records}
    folder = root / 'figures'
    folder.mkdir(parents=True, exist_ok=True)
    paths, output = [], []
    for letter, panel in zip('abcdefgh', spec['panels']):
        fig, ax = plt.subplots(figsize=spec.get('figsize', (6, 4.5)))
        curves = []
        for curve in panel['curves']:
            sources = [lookup[name] for name in curve['sources']]
            batch_sizes = sources[0]['batch_sizes']
            per_dataset = np.stack([np.array([np.mean(r['methods'][curve.get('method', 'promptmia')][str(b)][curve.get('metric', 'asr')])
                                              for b in batch_sizes])*100 for r in sources])
            mean = per_dataset.mean(0)
            ax.plot(batch_sizes, mean, marker=curve.get('marker', 'o'), ls=curve.get('ls', '-'), label=curve['label'])
            curves.append(dict(curve, batch_sizes=batch_sizes, mean=mean.tolist(), std=per_dataset.std(0).tolist(),
                               source_means=per_dataset.tolist()))
        ax.set_title(panel['title'])
        ax.set_xlabel(spec.get('xlabel', 'Batch size'))
        ax.set_ylabel('Attack Success Rate (%)')
        ax.grid(True)
        ax.legend(**{'fontsize': 9, **panel.get('legend', {})})
        if spec.get('tight_layout', True):
            fig.tight_layout()
        stem = folder / (spec['id']+(letter if len(spec['panels']) > 1 else ''))
        fig.savefig(stem.with_suffix('.pdf'), bbox_inches='tight')
        fig.savefig(stem.with_suffix('.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        paths += [stem.with_suffix('.png'), stem.with_suffix('.pdf')]
        output.append(dict(title=panel['title'], curves=curves))
    summary = folder / (spec['id']+'.json')
    _write_json(summary, dict(id=spec['id'], sources=spec.get('sources', []), panels=output))
    return paths+[summary]


def _detector_panels(spec, records, root, plt):
    """One image per detector (figure15a, ...): the t-SNE of the key pool, blue benign keys, red injected keys, and a
    cross on every key the detector flags."""
    lookup = {r['name']: r for r in records}
    folder = root / 'figures'
    folder.mkdir(parents=True, exist_ok=True)
    paths, output = [], []
    for letter, panel in zip('abcdefgh', spec['panels']):
        record = lookup[panel['sources'][0]]
        coordinates = np.asarray(record['coordinates'])
        injected = list(record.get('injected_slots', []))
        flagged = record['flagged'][panel['detector']]
        benign = [i for i in range(len(coordinates)) if i not in injected]
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(coordinates[benign, 0], coordinates[benign, 1], c='#9999ff', s=90, label='Benign Keys')
        if injected:
            ax.scatter(coordinates[injected, 0], coordinates[injected, 1], c='red', edgecolors='black', s=90,
                       label='Adversarial Keys', zorder=5)
        ax.scatter(coordinates[flagged, 0], coordinates[flagged, 1], marker='x', c='black', s=160, linewidths=2.5,
                   label='Flagged as outlier', zorder=6)
        ax.grid(True)
        ax.legend(fontsize=9)
        fig.tight_layout()
        stem = folder / (spec['id']+letter)
        fig.savefig(stem.with_suffix('.pdf'))
        fig.savefig(stem.with_suffix('.png'), dpi=150)
        plt.close(fig)
        paths += [stem.with_suffix('.png'), stem.with_suffix('.pdf')]
        output.append(dict(detector=panel['detector'], flagged=flagged, injected=injected,
                           true_positives=len(set(flagged) & set(injected)),
                           false_positives=len(set(flagged) - set(injected))))
    summary = folder / (spec['id']+'.json')
    _write_json(summary, dict(id=spec['id'], sources=spec.get('sources', []), panels=output))
    return paths+[summary]


def _gaussian_grid(spec, records, root, plt):
    """A grid of Gaussian query model panels: queries coloured by the nearest K-Means centre of the keys, and each key a
    2-D Gaussian whose scale is the MAD of the distances of the queries nearest to it; shading is the mixture log-density."""
    from sklearn.cluster import KMeans
    lookup = {r['name']: r for r in records}
    panels = spec['panels']
    ncols = int(spec['ncols'])
    fig, axes = plt.subplots(math.ceil(len(panels)/ncols), ncols, figsize=(18, 13), squeeze=False)
    for ax, panel in zip(axes.flat, panels):
        record = lookup[panel['sources'][0]]
        coordinates = np.asarray(record['coordinates'], dtype=np.float32)
        keys, queries = coordinates[:record['n_keys']], coordinates[record['n_keys']:]
        count = int(panel['n_clusters'])
        centers = KMeans(n_clusters=count, n_init=10, random_state=42).fit(keys).cluster_centers_
        regions = ((queries[:, None]-centers[None])**2).sum(-1).argmin(1)
        low, high = np.vstack([keys, queries]).min(0)-5, np.vstack([keys, queries]).max(0)+5
        xx, yy = np.meshgrid(np.linspace(low[0], high[0], 400), np.linspace(low[1], high[1], 400))
        grid = np.c_[xx.ravel(), yy.ravel()]
        distances = ((queries[:, None]-keys[None])**2).sum(-1)**0.5
        closest = distances.argmin(1)
        sigma = np.zeros(len(keys))
        for i in range(len(keys)):
            d = distances[closest == i, i]
            if len(d):
                mad = np.median(np.abs(d-np.median(d)))
                sigma[i] = max(mad/0.6745 if mad > 0 else d.std(), 1e-3)
        density = np.full(len(grid), -np.inf)
        for (x, y), s in zip(keys, sigma):
            if s > 0:
                density = np.logaddexp(density, -np.log(2*np.pi*s*s)-0.5*((grid[:, 0]-x)**2+(grid[:, 1]-y)**2)/(s*s))
        ax.contourf(xx, yy, density.reshape(xx.shape), levels=18, alpha=0.2)
        for c in range(count):
            members = regions == c
            ax.scatter(queries[members, 0], queries[members, 1], color=CLUSTER_COLORS[c], edgecolors='black', s=40,
                       label=f'Cluster {c+1} queries')
        ax.scatter(keys[:, 0], keys[:, 1], c='blue', edgecolors='black', s=60, label='Benign Keys', zorder=10)
        ax.set_title(panel['title'], fontsize=13)
        ax.grid(True)
        ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0)
    for ax in list(axes.flat)[len(panels):]:
        ax.set_visible(False)
    fig.tight_layout()
    stem = root / 'figures' / spec['id']
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix('.pdf'))
    fig.savefig(stem.with_suffix('.png'), dpi=100)
    plt.close(fig)
    _write_json(stem.with_suffix('.json'), dict(id=spec['id'], sources=spec.get('sources', []), panels=panels))
    return [stem.with_suffix('.png'), stem.with_suffix('.pdf'), stem.with_suffix('.json')]


def _save_panel(fig, stem):
    """One file per panel, at the size and resolution the paper's subfigures use."""
    fig.tight_layout()
    fig.savefig(stem.with_suffix('.pdf'))
    fig.savefig(stem.with_suffix('.png'), dpi=150)
    return [stem.with_suffix('.png'), stem.with_suffix('.pdf')]


def _key_panel(record, title, stem, plt):
    import matplotlib.patheffects as effects
    # t-SNE produced float32 values; keep that precision for identical drawing.
    keys = np.asarray(record['coordinates'], dtype=np.float32)
    injected = record['injected_slots']
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(keys[:, 0], keys[:, 1], c='blue', alpha=0.3, s=120, label='Benign Keys', zorder=5)
    if injected:
        ax.scatter(keys[injected, 0], keys[injected, 1], c='red', edgecolors='black', s=120, label='Adversarial Keys', zorder=10)
    for i in range(len(keys)):
        ax.text(*keys[i], str(i), fontsize=12, ha='center', va='bottom', zorder=30,
                fontweight='bold' if i in injected else 'normal', color='darkred' if i in injected else 'navy',
                path_effects=[effects.withStroke(linewidth=2, foreground='white', alpha=0.85)])
    ax.set_title(title, fontsize=16)
    ax.legend(fontsize=16, markerscale=1.5)
    ax.tick_params(labelsize=14)
    ax.grid(True)
    paths = _save_panel(fig, stem)
    plt.close(fig)
    return paths


def _query_panel(record, panel, stem, plt):
    from matplotlib.colors import ListedColormap
    from sklearn.cluster import KMeans
    coordinates = np.asarray(record['coordinates'], dtype=np.float32)
    keys, queries = coordinates[:record['n_keys']], coordinates[record['n_keys']:]
    mode = panel['mode']
    fig, ax = plt.subplots(figsize=(10, 8))
    if mode == 'scatter':
        ax.scatter(keys[:, 0], keys[:, 1], c='blue', alpha=0.3, s=60, label='Benign Keys', zorder=5)
        ax.scatter(queries[:, 0], queries[:, 1], c='green', edgecolors='black', s=60, label='Query Vectors', zorder=20)
    else:
        count = int(panel['n_clusters'])
        centers = KMeans(n_clusters=count, n_init=10, random_state=42).fit(keys).cluster_centers_
        regions = np.square(queries[:, None]-centers[None]).sum(-1).argmin(1)
        low, high = np.vstack([keys, queries]).min(0)-5, np.vstack([keys, queries]).max(0)+5
        xx, yy = np.meshgrid(np.linspace(low[0], high[0], 400), np.linspace(low[1], high[1], 400))
        grid = np.c_[xx.ravel(), yy.ravel()]
        palette = ListedColormap(['#ffb347', '#c9e4ff', '#fff3c4', '#d6d6d6', '#e6d6ff'])
        if mode == 'kmeans':
            ax.pcolormesh(xx, yy, np.square(grid[:, None]-centers[None]).sum(-1).argmin(1).reshape(xx.shape),
                          cmap=palette, alpha=0.25, shading='auto', zorder=0)
        else:
            distances = np.square(queries[:, None]-keys[None]).sum(-1)**0.5
            closest = distances.argmin(1)
            density = np.full(len(grid), -np.inf)
            for i, (x, y) in enumerate(keys):
                d = distances[closest == i, i]
                if len(d):
                    mad = np.median(np.abs(d-np.median(d)))
                    sigma = max(mad/0.6745 if mad > 0 else d.std(), 1e-3)
                    density = np.logaddexp(density, -np.log(2*np.pi*sigma*sigma)
                                           - 0.5*((grid[:, 0]-x)**2+(grid[:, 1]-y)**2)/(sigma*sigma))
            ax.contourf(xx, yy, density.reshape(xx.shape), levels=18, alpha=0.2, zorder=1)
        for c in range(count):
            members = regions == c
            ax.scatter(queries[members, 0], queries[members, 1], color=palette.colors[c], edgecolors='black',
                       linewidths=0.6, s=60, label=f'Cluster {c} queries')
        if mode == 'kmeans':
            ax.scatter(keys[:, 0], keys[:, 1], c='blue', alpha=0.6, edgecolors='black', s=60, label='Benign Keys', zorder=10)
            ax.scatter(centers[:, 0], centers[:, 1], facecolors='none', edgecolors='black', marker='D', s=200,
                       linewidths=1.2, zorder=12)
        else:
            ax.scatter(keys[:, 0], keys[:, 1], c='blue', edgecolors='black', linewidths=1.0, s=60, label='Benign Keys', zorder=10)
    ax.set_title(panel['title'], fontsize=20)
    ax.legend(fontsize=20 if mode == 'scatter' else 14, markerscale=1.5)
    ax.tick_params(labelsize=18)
    ax.grid(True)
    paths = _save_panel(fig, stem)
    plt.close(fig)
    return paths


def _averaged_curves(spec, records, root, plt):
    """One panel of curves averaged over datasets, sized to sit next to Figure 5 in the paper."""
    panel = spec['panels'][0]
    lookup = {r['name']: r for r in records}
    font = 12
    fig, ax = plt.subplots(figsize=(6.3, 4.0))
    curves = []
    for curve in panel['curves']:
        summary = summarize_attack([lookup[name] for name in curve['sources']],
                                   curve.get('method', 'promptmia'), curve.get('metric', 'asr'))
        x = np.arange(len(summary['batch_sizes']))
        ax.plot(x, np.asarray(summary['mean'])*100, marker='o', ms=4, lw=1.8, ls='-',
                color=curve['color'], label=curve['label'])
        curves.append(dict(curve, **summary))
    ax.set_xticks(x)
    ax.set_xticklabels(summary['batch_sizes'])
    ax.set_ylim(*panel['ylim'])
    ax.set_title(panel['title'], fontweight='bold', fontsize=font)
    ax.set_xlabel('Batch Size', fontsize=font)
    ax.set_ylabel('Attack Success Rate (%)', fontsize=font)
    ax.tick_params(labelsize=font)
    ax.grid(True, alpha=.4)
    ax.legend(fontsize=font-1, loc='lower left')
    stem = root / 'figures' / spec['id']
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = _save_panel(fig, stem)
    plt.close(fig)
    _write_json(stem.with_suffix('.json'), dict(id=spec['id'], sources=spec.get('sources', []),
                                                panels=[dict(title=panel['title'], curves=curves)]))
    return paths + [stem.with_suffix('.json')]


def _banded_curves(spec, records, root, plt):
    """One panel of curves averaged over datasets, with a band of one std across the datasets."""
    panel = spec['panels'][0]
    lookup = {r['name']: r for r in records}
    fig, ax = plt.subplots(figsize=(10, 6))
    curves = []
    for curve in panel['curves']:
        summary = summarize_attack([lookup[name] for name in curve['sources']],
                                   curve.get('method', 'promptmia'), curve.get('metric', 'asr'))
        x = np.arange(len(summary['batch_sizes']))
        mean, std = np.asarray(summary['mean'])*100, np.asarray(summary['std'])*100
        ax.plot(x, mean, marker=curve['marker'], linewidth=2.5, color=curve['color'], label=curve['label'])
        ax.fill_between(x, mean-std, mean+std, color=curve['color'], alpha=.2)
        curves.append(dict(curve, **summary))
    ax.set_xticks(x)
    ax.set_xticklabels(summary['batch_sizes'])
    ax.set_ylim(*panel['ylim'])
    ax.set_xlabel('Batch Size', fontsize=18)
    ax.set_ylabel('Attack Success Rate (%)', fontsize=18)
    ax.set_title(panel['title'], fontsize=20)
    ax.grid(True, linestyle='--', alpha=.4)
    ax.legend(fontsize=14)
    stem = root / 'figures' / spec['id']
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = _save_panel(fig, stem)
    plt.close(fig)
    _write_json(stem.with_suffix('.json'), dict(id=spec['id'], sources=spec.get('sources', []),
                                                panels=[dict(title=panel['title'], curves=curves)]))
    return paths + [stem.with_suffix('.json')]


def _dataset_panels(spec, records, root, plt):
    """One row of per-dataset panels with an error band over the game repeats, printed at the full text width."""
    lookup = {r['name']: r for r in records}
    font = 18
    fig, axes = plt.subplots(1, len(spec['panels']), figsize=(15, 4.2))
    output = []
    for i, (ax, panel) in enumerate(zip(axes, spec['panels'])):
        result = dict(title=panel['title'], curves=[])
        for curve in panel['curves']:
            summary = summarize_attack([lookup[name] for name in curve['sources']],
                                       curve.get('method', 'promptmia'), curve.get('metric', 'asr'))
            x = np.arange(len(summary['batch_sizes']))
            mean, std = np.asarray(summary['mean'])*100, np.asarray(summary['std'])*100
            ax.plot(x, mean, marker='o', ms=5, lw=2, color=curve['color'], label=curve['label'])
            ax.fill_between(x, mean-std, mean+std, color=curve['color'], alpha=.2)
            ax.set_xticks(x)
            ax.set_xticklabels(summary['batch_sizes'])
            result['curves'].append(dict(curve, **summary))
        ax.set_ylim(*panel['ylim'])
        ax.set_title(panel['title'], fontsize=font)
        ax.set_xlabel('Batch Size', fontsize=font)
        if i == 0:
            ax.set_ylabel('ASR (%)', fontsize=font)
        ax.tick_params(labelsize=font-4)
        ax.tick_params(axis='x', labelsize=font-6)
        ax.grid(True, alpha=.4)
        ax.legend(fontsize=font-4, loc='lower left')
        output.append(result)
    fig.suptitle(spec['title'], fontsize=font)
    stem = root / 'figures' / spec['id']
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = _save_panel(fig, stem)
    plt.close(fig)
    _write_json(stem.with_suffix('.json'), dict(id=spec['id'], sources=spec.get('sources', []), panels=output))
    return paths + [stem.with_suffix('.json')]


def _panel_files(spec, records, root, plt):
    """Figures 2 and 3 are one image per subfigure: figure2a, figure2b, ..."""
    lookup = {r['name']: r for r in records}
    folder = root / 'figures'
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for letter, panel in zip('abcdefgh', spec['panels']):
        record = lookup[panel['sources'][0]]
        stem = folder / (spec['id']+letter)
        if panel['mode'] == 'keys':
            paths += _key_panel(record, panel['title'], stem, plt)
        else:
            paths += _query_panel(record, panel, stem, plt)
    summary = folder / (spec['id']+'.json')
    _write_json(summary, dict(id=spec['id'], sources=spec.get('sources', []), panels=spec['panels']))
    return paths+[summary]


def _diagram(spec, plt):
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.8)
    ax.axis('off')
    def box(x, y, w, h, label, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.12', facecolor=color, edgecolor='#344054'))
        ax.text(x+w/2, y+h/2, label, ha='center', va='center', fontsize=11)
    def arrow(start, end, label=''):
        ax.annotate('', xy=end, xytext=start, arrowprops=dict(arrowstyle='->', color='#344054', lw=1.8))
        if label:
            ax.text((start[0]+end[0])/2, (start[1]+end[1])/2+.12, label, ha='center', fontsize=9)
    if spec['id'] == 'figure1':
        box(.4, 1.2, 2.1, 2.1, 'Server\nTarget query q(T)\nAdversarial keys', '#ffe4e4')
        box(3.6, 1.2, 2.1, 2.1, 'Client\nPrivate batch\nTop-N selection', '#e5efff')
        box(6.8, 1.2, 2.1, 2.1, 'Local training\nUpdate selected\nprompts and keys', '#e5efff')
        box(10, 1.2, 1.6, 2.1, 'Server\nMembership\ndecision', '#ffe4e4')
        arrow((2.6, 2.3), (3.5, 2.3), '1. Inject')
        arrow((5.8, 2.3), (6.7, 2.3), '2. Train')
        arrow((9, 2.3), (9.9, 2.3), '3. Return')
        ax.text(6, .35, 'All injected prompts updated → predict member', ha='center', fontsize=12)
    else:
        box(4.5, 3.35, 3, .9, 'Global key–prompt pool', '#e5efff')
        for x, name in [(1, 'Client 1'), (4.5, 'Client 2'), (8, 'Client C')]:
            box(x, 1, 3, 1.2, name+'\nFrozen backbone + selected prompts\nLocal private data', '#e8f5e9')
            arrow((6, 3.2), (x+1.5, 2.35))
        box(4.5, .05, 3, .55, 'Aggregate client updates', '#fff2cc')
        for x in (2.5, 6, 9.5):
            arrow((x, .9), (6, .7))
        ax.text(11.75, 3.7, 'Repeat each round', ha='right', fontsize=10)
    fig.tight_layout()
    return fig, dict(description='PromptMIA workflow' if spec['id'] == 'figure1' else 'Federated prompt tuning workflow')


def _final_accuracy(record, key='test_acc'):
    values = record['metrics'][key]
    last = max(map(int, values))
    return float(values[str(last)])


def _mean_std(values, ddof=1):
    values = np.asarray(values, dtype=float)
    if not len(values):
        raise ValueError('A table entry is missing its measurement files')
    return float(values.mean()), float(values.std(ddof=ddof) if len(values) > ddof else 0.)


def _table_rows(spec, records):
    number = int(spec['id'].removeprefix('table'))
    if number in (1, 6):
        methods = ('IsolationForest', 'LocalOutlierFactor', 'OneClassSVM', 'EllipticEnvelope')
        if number == 1:
            rows = []
            for method in methods:
                row = dict(method=method)
                for metric in ('precision', 'recall', 'f1'):
                    mean, std = _mean_std([r['methods'][method][metric] for r in records], ddof=0)
                    row[metric], row[metric+'_std'] = mean, std
                rows.append(row)
            return rows
        return [dict(dataset=r['dataset'], model=r['backbone'], method=m,
                     **{k: r['methods'][m][k] for k in ('precision', 'recall', 'f1')})
                for r in records for m in methods]
    if number == 2:
        rows = []
        for dataset in DATASETS:
            source = [r for r in records if r['dataset'] == dataset]
            if not source:
                continue
            row = dict(dataset=dataset)
            for name, epsilon in [('eps3', 3), ('eps5', 5), ('eps8', 8), ('non_dp', None)]:
                values = [100*_final_accuracy(r, spec.get('accuracy_key', 'test_acc')) for r in source
                          if r['config'].get('dp_eps') == epsilon]
                row[name], row[name+'_std'] = _mean_std(values, int(spec.get('ddof', 1)))
                row[name+'_runs'] = len(values)
            rows.append(row)
        return rows
    if number == 3:
        return [dict(number=row.get('number', index+1), **{
                    key: row[key] for key in ('attack', 'surface', 'methodology', 'assumptions')})
                for index, row in enumerate(spec['rows'])]
    if number == 4:
        rows = []
        for rate in ('1/10', '1/60'):
            row = dict(injection_rate=rate)
            for c in (1, 2, 3):
                row[f'c{c}'], row[f'c{c}_std'] = _mean_std(
                    [r['pooled'][rate][str(c)]['f1'] for r in records], ddof=0)
            rows.append(row)
        return rows
    if number == 5:
        rows = []
        for pool_size in sorted({int(r['config']['M']) for r in records}):
            row = dict(M=pool_size)
            for dataset in ('cifar100', 'tinyimagenet'):
                selected = [r for r in records if r['dataset'] == dataset and int(r['config']['M']) == pool_size]
                for batch in (64, 256):
                    if not selected:
                        raise ValueError(f'Table 5 is missing {dataset}, M={pool_size}')
                    row[f'{dataset}_B{batch}'] = 100*float(np.mean([
                        np.mean(r['methods']['promptmia'][str(batch)]['asr']) for r in selected]))
            rows.append(row)
        return rows
    if number == 7:
        rows = []
        for index in range(5):
            row = dict(group=('Q1 (most distinctive)' if index == 0 else
                              'Q5 (least distinctive)' if index == 4 else f'Q{index+1}'))
            for batch in (64, 256):
                for metric in ('fpr', 'asr'):
                    row[f'{metric}_B{batch}'] = 100*float(np.mean([
                        r['quintiles'][str(batch)][index][metric] for r in records]))
            rows.append(row)
        return rows
    if number == 8:
        rows = []
        groups = sorted({(r['dataset'], r['backbone']) for r in records})
        for dataset, backbone in groups:
            source = [r for r in records if r['dataset'] == dataset and r['backbone'] == backbone]
            clean = [_final_accuracy(r) for r in source if not r['config'].get('inject_round', 0)]
            injected = [_final_accuracy(r) for r in source if r['config'].get('inject_round', 0)]
            cm, cs = _mean_std(clean, int(spec.get('ddof', 1)))
            im, ins = _mean_std(injected, int(spec.get('ddof', 1)))
            rows.append(dict(dataset=dataset, model=backbone, clean_accuracy=cm, clean_std=cs,
                             injected_accuracy=im, injected_std=ins, clean_runs=len(clean), injected_runs=len(injected)))
        return rows
    if number == 9:
        return [row for result in records for row in result['rows']]
    raise ValueError(f'Unknown table: {spec["id"]}')


def _latex(text):
    return str(text).replace('\\', r'\textbackslash{}').replace('&', r'\&').replace('%', r'\%').replace('_', r'\_')


def _save_table(spec, rows, root):
    stem = root / 'tables' / spec['id']
    stem.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0])
    with stem.with_suffix('.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    _write_json(stem.with_suffix('.json'), dict(id=spec['id'], sources=spec.get('sources', []), rows=rows))
    n = int(spec['id'].removeprefix('table'))
    if n == 1:
        headings = ['Method', 'Precision', 'Recall', 'F1']
        cells = [[r['method']]+[f"{r[m]:.2f} ({r[m+'_std']:.2f})" for m in ('precision', 'recall', 'f1')] for r in rows]
    elif n == 2:
        headings = ['Dataset', r'$\epsilon=3$', r'$\epsilon=5$', r'$\epsilon=8$', 'Non-DP']
        cells = [[DATASET_LABELS[r['dataset']]]+[f"{r[m]:.1f} ({r[m+'_std']:.1f})" for m in ('eps3', 'eps5', 'eps8', 'non_dp')] for r in rows]
    elif n == 3:
        headings = [r'\#', 'Attack', 'Attack surface', 'Methodology', 'Assumptions']
        cells = [[r[c] for c in columns] for r in rows]
    elif n == 4:
        headings = ['Injection rate', '$c=1$', '$c=2$', '$c=3$']
        cells = [[r['injection_rate']]+[f'{r[f"c{c}"]:.3f}' for c in (1, 2, 3)] for r in rows]
    elif n == 5:
        headings = ['$M$', 'CIFAR-100 $B=64$', 'CIFAR-100 $B=256$', 'TinyImageNet $B=64$', 'TinyImageNet $B=256$']
        cells = [[str(r['M'])]+[f'{r[f"{d}_B{b}"]:.1f}' for d in ('cifar100', 'tinyimagenet')
                               for b in (64, 256)] for r in rows]
    elif n == 6:
        headings = ['Dataset', 'Model', 'Method', 'Precision', 'Recall', 'F1']
        cells = [[DATASET_LABELS[r['dataset']], r['model'], r['method']]+[f'{r[m]:.4f}' for m in ('precision', 'recall', 'f1')] for r in rows]
    elif n == 7:
        headings = ['Group', 'FPR $B=64$', 'ASR $B=64$', 'FPR $B=256$', 'ASR $B=256$']
        cells = [[r['group']]+[f'{r[f"{m}_B{b}"]:.2f}' for b in (64, 256) for m in ('fpr', 'asr')] for r in rows]
    elif n == 8:
        headings = ['Dataset', 'Model', 'Model accuracy', 'Model accuracy with PromptMIA']
        cells = [[DATASET_LABELS[r['dataset']], r['model'], f"{r['clean_accuracy']:.4f}",
                  f"{r['injected_accuracy']:.4f}"] for r in rows]
    elif n == 9:
        headings = ['Model', 'Total params', 'Modified params', r'Sparsity (\%)']
        cells = [[r['model'], f"{r['total_params']:,}", f"{r['modified_params']:,}", f"{r['sparsity_percent']:.4f}"] for r in rows]
    else:
        raise ValueError(f'Unknown table: {spec["id"]}')
    alignment = 'rp{.15\\linewidth}p{.18\\linewidth}p{.28\\linewidth}p{.24\\linewidth}' if n == 3 else 'l'+'r'*(len(headings)-1)
    lines = [r'\begin{table}[htbp]', r'\centering', '\\begin{tabular}{'+alignment+'}', r'\toprule',
             ' & '.join(headings)+r' \\', r'\midrule']
    lines += [' & '.join(_latex(c) if '$' not in str(c) else str(c) for c in row)+r' \\' for row in cells]
    lines += [r'\bottomrule', r'\end{tabular}', '\\caption{'+_latex(spec.get('title', spec['id']))+'}',
              '\\label{tab:'+spec['id']+'}', r'\end{table}']
    stem.with_suffix('.tex').write_text('\n'.join(lines)+'\n')
    return [stem.with_suffix(s) for s in ('.json', '.csv', '.tex')]


def _round_panels(spec, records, root, plt):
    """One untitled file per training round (figure18a, b, ...): t-SNE of the benign keys and the query vectors."""
    lookup = {r['name']: r for r in records}
    folder = root / 'figures'
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for letter, panel in zip('abcdefgh', spec['panels']):
        record = lookup[panel['sources'][0]]
        coordinates = np.asarray(record['coordinates'], dtype=np.float32)
        keys, queries = coordinates[:record['n_keys']], coordinates[record['n_keys']:]
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(queries[:, 0], queries[:, 1], c='darkgreen', s=20, label='Query Vectors')
        ax.scatter(keys[:, 0], keys[:, 1], c='blue', alpha=0.5, s=40, label='Benign Keys', zorder=5)
        ax.grid(True)
        ax.legend(fontsize=9)
        paths += _save_panel(fig, folder / (spec['id']+letter))
        plt.close(fig)
    summary = folder / (spec['id']+'.json')
    _write_json(summary, dict(id=spec['id'], sources=spec.get('sources', []), panels=spec['panels']))
    return paths+[summary]


def _attack_row(spec, records, root, plt):
    """One row of panels of Advantage and ASR against batch size: the mean over the panel's runs of each run's mean
    over the game repeats; a single-run panel can add a band of one std over the repeats."""
    lookup = {r['name']: r for r in records}
    fig, axes = plt.subplots(1, len(spec['panels']), figsize=spec['figsize'])
    output = []
    for ax, panel in zip(axes, spec['panels']):
        curves = []
        for method in panel['methods']:
            for metric in panel['metrics']:
                summary = summarize_attack([lookup[name] for name in panel['sources']], method, metric)
                # markers near 100% stay whole at the top of the axis
                mean, style = 100*np.asarray(summary['mean']), COMPARISON_STYLE[(method, metric)]
                ax.plot(summary['batch_sizes'], mean, clip_on=False, zorder=3, **style)
                if spec.get('bands'):
                    std = 100*np.asarray(summary['std'])
                    ax.fill_between(summary['batch_sizes'], mean-std, mean+std, color=style['color'], alpha=0.2)
                curves.append(dict(method=method, metric=metric, **summary))
        ax.set_title(panel['title'])
        ax.set_xlabel('Batch Size')
        ax.set_ylabel('Percentage (%)')
        ax.set_ylim(*panel['ylim'])
        ax.grid(True)
        output.append(dict(title=panel['title'], sources=panel['sources'], curves=curves))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(handles), bbox_to_anchor=(0.5, spec['legend_y']))
    if spec.get('suptitle'):
        fig.suptitle(spec['suptitle'])
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    stem = root / 'figures' / spec['id']
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(stem.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    _write_json(stem.with_suffix('.json'), dict(id=spec['id'], sources=spec.get('sources', []), panels=output))
    return [stem.with_suffix('.png'), stem.with_suffix('.pdf'), stem.with_suffix('.json')]


def plot_artifact(spec, root):
    root = Path(root).resolve()
    records = [_read(source, root) for source in spec.get('sources', [])]
    if spec['id'].startswith('table'):
        return _save_table(spec, _table_rows(spec, records), root)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    number = int(spec['id'].removeprefix('figure'))
    if number in (2, 3):
        return _panel_files(spec, records, root, plt)
    if number == 6:
        return _averaged_curves(spec, records, root, plt)
    if number == 7:
        return _dataset_panels(spec, records, root, plt)
    if number == 9:
        return _banded_curves(spec, records, root, plt)
    if number == 10:
        return _bound_panels(spec, records, root, plt)
    if number == 11:
        return _gaussian_grid(spec, records, root, plt)
    if number in (12, 13, 14):
        return _comparison_row(spec, records, root, plt)
    if number in (15, 16):
        return _detector_panels(spec, records, root, plt)
    if number in (17, 21):
        return _dataset_mean_panels(spec, records, root, plt)
    if number == 18:
        return _round_panels(spec, records, root, plt)
    if number in (19, 20):
        return _attack_row(spec, records, root, plt)
    if spec.get('static_asset'):
        source = root / spec['static_asset']
        pixels = plt.imread(source)
        fig = plt.figure(figsize=(pixels.shape[1]/180, pixels.shape[0]/180))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(pixels)
        ax.axis('off')
        summary = dict(static_asset=spec['static_asset'])
    elif number in (1, 8):
        fig, summary = _diagram(spec, plt)
    elif number == 5:
        fig, summary = _baseline_figure(spec, records, plt)
    else:
        fig, summary = _line_figure(spec, records, plt)
    stem = root / 'figures' / spec['id']
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in ('.png', '.pdf'):
        path = stem.with_suffix(suffix)
        if suffix == '.png' and spec.get('static_asset'):
            shutil.copyfile(root / spec['static_asset'], path)
        else:
            fig.savefig(path, dpi=150 if number == 5 else 180, bbox_inches='tight', pad_inches=0 if spec.get('static_asset') else .1)
        paths.append(path)
    plt.close(fig)
    _write_json(stem.with_suffix('.json'), dict(id=spec['id'], sources=spec.get('sources', []), **summary))
    paths.append(stem.with_suffix('.json'))
    return paths
