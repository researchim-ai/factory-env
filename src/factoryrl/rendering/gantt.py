"""Dependency-free SVG Gantt chart embedded in a small HTML document."""

from __future__ import annotations

import html
from typing import Any


def render_gantt_html(intervals: list[dict[str, Any]], machine_ids: list[str]) -> str:
    """Draw completed intervals. Empty schedules still return a valid document."""
    machines = list(machine_ids)
    for interval in intervals:
        if interval["machine_id"] not in machines:
            machines.append(str(interval["machine_id"]))
    horizon = max((float(interval["end"]) for interval in intervals), default=1.0)
    horizon = max(horizon, 1.0)
    width = 840
    row_h = 36
    left = 90
    top = 30
    height = top + row_h * max(len(machines), 1) + 20
    plot_w = width - left - 20
    colors = ("#1b6ca8", "#c46b2c", "#2f7d4a", "#6b4c9a", "#a33b4a", "#3d7a7a")
    job_color: dict[str, str] = {}
    parts = [
        "<!DOCTYPE html>",
        '<html><head><meta charset="utf-8"><title>FactoryRL Gantt</title></head>',
        "<body>",
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        'font-family="sans-serif" font-size="12">',
        '<rect width="100%" height="100%" fill="#f7f5f2"/>',
    ]
    for index, machine_id in enumerate(machines):
        y = top + index * row_h
        parts.append(
            f'<text x="8" y="{y + 22}">{html.escape(machine_id)}</text>'
            f'<line x1="{left}" y1="{y + row_h - 6}" x2="{left + plot_w}" '
            f'y2="{y + row_h - 6}" stroke="#ccc"/>'
        )
    for interval in intervals:
        machine_id = str(interval["machine_id"])
        row = machines.index(machine_id)
        start = float(interval["start"])
        end = float(interval["end"])
        x = left + plot_w * (start / horizon)
        w = max(2.0, plot_w * ((end - start) / horizon))
        y = top + row * row_h + 4
        job_id = str(interval.get("job_id", ""))
        if job_id not in job_color:
            job_color[job_id] = colors[len(job_color) % len(colors)]
        label = html.escape(str(interval["operation_id"]))
        parts.append(
            f'<rect x="{x:.2f}" y="{y}" width="{w:.2f}" height="{row_h - 12}" '
            f'fill="{job_color[job_id]}" rx="3"/>'
            f'<text x="{x + 4:.2f}" y="{y + 16}" fill="white">{label}</text>'
        )
    parts.append("</svg></body></html>")
    return "\n".join(parts)
