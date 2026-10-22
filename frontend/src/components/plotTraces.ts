/** Build a mean line plus a ±1 s.d. shaded band for one method. */
export function bandTraces(
  x: number[],
  mean: number[],
  std: number[],
  name: string,
  color: string,
): Plotly.Data[] {
  const upper = mean.map((m, i) => m + std[i]);
  const lower = mean.map((m, i) => m - std[i]);
  const rgba = hexToRgba(color, 0.15);
  return [
    {
      x: [...x, ...[...x].reverse()],
      y: [...upper, ...[...lower].reverse()],
      fill: 'toself',
      fillcolor: rgba,
      line: { color: 'transparent' },
      hoverinfo: 'skip',
      showlegend: false,
      type: 'scatter',
    },
    {
      x,
      y: mean,
      name,
      mode: 'lines',
      line: { color, width: 2.4, shape: 'hv' },
      type: 'scatter',
      hovertemplate: `<b>${name}</b><br>%{x} measurements<br>best = %{y:.3f}<extra></extra>`,
    },
  ];
}

function hexToRgba(hex: string, alpha: number): string {
  const h = hex.replace('#', '');
  const r = parseInt(h.slice(0, 2), 16);
  const g = parseInt(h.slice(2, 4), 16);
  const b = parseInt(h.slice(4, 6), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}
