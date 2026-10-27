from base64 import b64decode
from pathlib import Path

from src.bench import BenchmarkLoader
from src.visual_variants import extent_for, render_background
from src.vlm_gate.utils.scatter_render import render_input_scatter
from src.vlm_gate.utils.task_input_adapter import _TaskInputDataset, _step_from_task
from src.agent_gate.tools.render_overlay import render_gate_overlay


CASES = [
    ('Acute2020', '994570_Normalized', 18),
    ('Bjornson', 'R15W11_W070515101208_Hu_M_Basal', 7),
    ('FRDR_covid19', '20200416p8_p040n02_C05_003', 14),
]
ROOT = Path('results/visual_luna_medium_definition/preview')
loader = BenchmarkLoader('benchmark', 'data')
for ds, sample, step_num in CASES:
    ti = loader.load_task(Path('benchmark') / ds / sample / f'step_{step_num:02d}' / 'task.json')
    if ti is None:
        raise RuntimeError(f'Missing task: {ds}/{sample}/step_{step_num:02d}')
    adapter = _TaskInputDataset(ti)
    step = _step_from_task(ti)
    x, y = adapter.get_xy_vals(step)
    extent = extent_for(x, y)
    output = ROOT / ds / sample / f'step_{step_num:02d}'
    output.mkdir(parents=True, exist_ok=True)
    xmin, xmax, ymin, ymax = extent
    gates = {'preview_box': {'x_min': xmin + .25 * (xmax - xmin),
                             'x_max': xmin + .75 * (xmax - xmin),
                             'y_min': ymin + .25 * (ymax - ymin),
                             'y_max': ymin + .75 * (ymax - ymin)}}
    for style in ('scatter', 'contour', 'density'):
        bg = output / f'{style}_bg.png'
        render_background(x, y, bg, extent, style)
        bare_uri = render_input_scatter(
            scatter_bg_path=str(bg), scatter_extent=extent,
            x_marker=ti.x_marker_display, y_marker=ti.y_marker_display,
            step_label=f'step_{step_num:02d}',
        )
        (output / f'{style}_vlm.png').write_bytes(b64decode(bare_uri.split(',', 1)[1]))
        overlay_uri, diagnostics = render_gate_overlay(
            gates, scatter_bg_path=str(bg), scatter_extent=extent,
            x_marker=ti.x_marker_display, y_marker=ti.y_marker_display,
            options=['preview_box'], step_label=f'step_{step_num:02d}',
        )
        (output / f'{style}_agent.png').write_bytes(b64decode(overlay_uri.split(',', 1)[1]))
        (output / f'{style}_diagnostics.txt').write_text(diagnostics)
        print(f'{ds}/{sample}/step_{step_num:02d} {style}: {len(x)} cells, extent={extent}')
