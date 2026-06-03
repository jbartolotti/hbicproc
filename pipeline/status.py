from pathlib import Path

from .pipeline import PipelineRunner
from .state import load_subject_state


def _find_subject_rows(config):
    bids_root = Path(config["bidskit"]["output_dir"])
    if not bids_root.exists():
        return []

    rows = []
    for subject_dir in sorted(p for p in bids_root.iterdir() if p.is_dir() and p.name.startswith("sub-")):
        sessions = sorted(p.name for p in subject_dir.iterdir() if p.is_dir() and p.name.startswith("ses-"))
        if sessions:
            for session in sessions:
                rows.append((subject_dir.name, session))
        else:
            rows.append((subject_dir.name, None))
    return rows


def _state_matrix(config, rows, stage_names):
    matrix = []
    for subject, session in rows:
        state = load_subject_state(Path(config["study_root"]) / subject)
        row = [bool(state.get(stage)) for stage in stage_names]
        matrix.append(row)
    return matrix


def _svg_color_cell(x, y, width, height, filled):
    fill = "#8fd19e" if filled else "#ffffff"
    stroke = "#888888"
    return f"<rect x=\"{x}\" y=\"{y}\" width=\"{width}\" height=\"{height}\" fill=\"{fill}\" stroke=\"{stroke}\"/>"


def _svg_text(x, y, text, size=12, anchor="start", weight="normal"):
    return f"<text x=\"{x}\" y=\"{y}\" font-size=\"{size}\" text-anchor=\"{anchor}\" font-weight=\"{weight}\">{text}</text>"


def _make_svg(rows, stage_names, matrix, output_path):
    row_height = 28
    label_width = 200
    cell_width = 120
    padding = 20
    num_rows = len(rows)
    width = label_width + cell_width * len(stage_names) + padding * 2
    height = padding * 2 + row_height * (num_rows + 2)

    title = "HBICProc Pipeline Status"
    header_y = padding + row_height
    body_start_y = header_y + row_height

    cells = []
    labels = []

    labels.append(_svg_text(padding, padding + 16, title, size=18, weight="bold"))
    labels.append(_svg_text(padding, header_y + 16, "Subject / Session", size=12, weight="bold"))
    for index, stage in enumerate(stage_names):
        x = padding + label_width + index * cell_width + cell_width / 2
        labels.append(_svg_text(x, header_y + 16, stage, size=12, anchor="middle", weight="bold"))

    for row_index, ((subject, session), row_values) in enumerate(zip(rows, matrix)):
        y = body_start_y + row_index * row_height
        label = f"{subject}" if session is None else f"{subject} | {session}"
        labels.append(_svg_text(padding + 4, y + 18, label, size=11, anchor="start"))
        for col_index, filled in enumerate(row_values):
            x = padding + label_width + col_index * cell_width
            cells.append(_svg_color_cell(x, y, cell_width, row_height, filled))

    legend_x = padding
    legend_y = body_start_y + num_rows * row_height + 16
    legend = [
        _svg_text(legend_x, legend_y, "Legend:", size=12, weight="bold"),
        _svg_color_cell(legend_x + 70, legend_y - 14, 18, 18, True),
        _svg_text(legend_x + 95, legend_y, "Complete", size=12, anchor="start"),
        _svg_color_cell(legend_x + 160, legend_y - 14, 18, 18, False),
        _svg_text(legend_x + 185, legend_y, "Incomplete", size=12, anchor="start"),
    ]

    svg = [
        f"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\">",
        f"<rect x=\"0\" y=\"0\" width=\"{width}\" height=\"{height}\" fill=\"#ffffff\"/>",
        *labels,
        *cells,
        *legend,
        "</svg>",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(svg), encoding="utf-8")
    return output_path


def save_pipeline_status_figure(config, output_path=None):
    rows = _find_subject_rows(config)
    stage_names = PipelineRunner(config).stage_order
    matrix = _state_matrix(config, rows, stage_names)

    if output_path:
        output_file = Path(output_path)
    else:
        output_file = Path(config["bidskit"]["output_dir"]) / "code" / "pipeline_status.svg"

    return _make_svg(rows, stage_names, matrix, output_file)
