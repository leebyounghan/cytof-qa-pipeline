"""Select deterministic cuts from the exact LLM-Gate prompt representation.

The shared rule table fixes the number of cuts needed per axis. When more
landmarks are available, select the deepest peak-to-valley separations;
when a three-region rule has only one valley, drop its middle region.
No expert labels enter cut selection.
With --otsu-fallback, an axis lacking enough valleys is split using its
stored 48-bin histogram; all required cuts on that axis are then chosen by
one- or two-threshold Otsu optimization. The existing valley-only behavior
is unchanged unless that option is supplied.
"""

import argparse
import json
from itertools import combinations
from pathlib import Path


def selected_valleys(axis_info: dict, needed: int) -> tuple[list[float], str]:
    if needed == 0:
        return [], 'none'
    peaks = axis_info['peaks']
    candidates = axis_info['valleys']
    scored = []
    for i, valley in enumerate(candidates):
        left = peaks[int(valley['left_peak_idx'])]['height']
        right = peaks[int(valley['right_peak_idx'])]['height']
        prominence = min(float(left), float(right)) - float(valley['depth'])
        scored.append((prominence, i, float(valley['x'])))
    scored.sort(key=lambda item: (-item[0], item[1]))
    cuts = sorted(value for _, _, value in scored[:needed])
    if len(cuts) == needed:
        return cuts, 'selected' if len(candidates) > needed else 'all'
    if needed == 2 and len(cuts) == 1:
        return cuts, 'middle_dropped'
    return cuts, 'insufficient'


def otsu_cuts(axis_info: dict, needed: int) -> list[float] | None:
    """Maximize between-region variance on stored histogram bins.

    Each region must contain positive count. A missing feasible partition
    is returned as None so the caller records a visible failed step.
    """
    if needed not in (1, 2):
        raise ValueError(f'Otsu fallback supports one or two cuts, got {needed}')
    centers = [float(x) for x in axis_info['bin_centers']]
    counts = [int(x) for x in axis_info['counts']]
    if len(centers) != len(counts) or len(centers) < needed + 1:
        raise ValueError('histogram centers/counts have incompatible lengths')
    if any(b <= a for a, b in zip(centers, centers[1:])):
        raise ValueError('histogram bin centers must be strictly increasing')
    if any(count < 0 for count in counts) or sum(counts) != axis_info['n']:
        raise ValueError('histogram counts must be nonnegative and sum to axis n')

    mass = [0]
    weighted = [0.0]
    for center, count in zip(centers, counts):
        mass.append(mass[-1] + count)
        weighted.append(weighted[-1] + center * count)

    best_boundaries = None
    best_score = float('-inf')
    for boundaries in combinations(range(1, len(counts)), needed):
        edges = (0, *boundaries, len(counts))
        score = 0.0
        for left, right in zip(edges, edges[1:]):
            region_mass = mass[right] - mass[left]
            if not region_mass:
                break
            region_weighted = weighted[right] - weighted[left]
            score += region_weighted * region_weighted / region_mass
        else:
            if score > best_score:
                best_score = score
                best_boundaries = boundaries
    if best_boundaries is None:
        return None
    return [(centers[index - 1] + centers[index]) / 2 for index in best_boundaries]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=Path, required=True)
    parser.add_argument('--benchmark', type=Path, default=Path('benchmark'))
    parser.add_argument('--otsu-fallback', action='store_true',
                        help='Use stored histogram counts when required KDE valleys are missing')
    args = parser.parse_args()
    count = 0
    for meta_path in sorted(args.results_dir.rglob('step_*/meta.json')):
        step_dir = meta_path.parent
        meta = json.loads(meta_path.read_text())
        if not meta['rule']['supported'] or meta['n_parent'] == 0:
            continue
        relative = step_dir.relative_to(args.results_dir)
        gate_path = args.benchmark / relative / 'gate_task.json'
        if not gate_path.exists():
            raise FileNotFoundError(f'{gate_path}: required LLM-Gate landmarks missing')
        gate = json.loads(gate_path.read_text())
        if gate['n_parent_cells'] != meta['n_parent']:
            raise ValueError(f'{gate_path}: parent size mismatch')
        if gate['step'] != meta['step'] or gate['cohort'] != meta['dataset']:
            raise ValueError(f'{gate_path}: task identity mismatch')
        selector = 'peak_valley_prominence_with_otsu_fallback' if args.otsu_fallback else 'peak_valley_prominence'
        result = {'source': str(gate_path), 'selector': selector}
        for axis in ('x', 'y'):
            axis_info = gate['axis_distribution'][axis]
            if axis_info['n'] != meta['n_parent']:
                raise ValueError(f'{gate_path}: {axis} axis parent size mismatch')
            needed = meta['rule'][f'n_{axis}_valleys']
            cuts, method = selected_valleys(axis_info, needed)
            if args.otsu_fallback and method in ('insufficient', 'middle_dropped'):
                fallback_cuts = otsu_cuts(axis_info, needed)
                if fallback_cuts is None:
                    cuts, method = [], 'otsu_infeasible'
                else:
                    cuts, method = fallback_cuts, 'otsu_fallback'
            result[axis] = cuts
            result[f'{axis}_method'] = method
            result[f'{axis}_available'] = len(axis_info['valleys'])
            if method in ('insufficient', 'otsu_infeasible'):
                result[f'{axis}_error'] = (f'{len(axis_info["valleys"])} landmarks for {needed} required cuts; '
                                            f'histogram Otsu partition {method}')
        (step_dir / 'valleys.json').write_text(json.dumps(result))
        count += 1
    if not count:
        raise RuntimeError(f'{args.results_dir}: no supported nonempty steps')
    print(f'kde_landmark_valleys: wrote {count} step files')


if __name__ == '__main__':
    main()
