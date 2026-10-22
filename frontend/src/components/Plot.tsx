import { useEffect, useRef } from 'react';
import Plotly from 'plotly.js-dist-min';

/** Light-theme defaults shared by every chart in the dashboard. */
const PLOT_LAYOUT: Partial<Plotly.Layout> = {
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor: 'rgba(0,0,0,0)',
  font: { color: '#59655f', family: 'Inter, system-ui, sans-serif', size: 12 },
  margin: { l: 58, r: 18, t: 28, b: 48 },
  xaxis: { gridcolor: 'rgba(34,58,45,0.09)', zerolinecolor: 'rgba(34,58,45,0.18)' },
  yaxis: { gridcolor: 'rgba(34,58,45,0.09)', zerolinecolor: 'rgba(34,58,45,0.18)' },
  legend: { bgcolor: 'rgba(0,0,0,0)', orientation: 'h', y: -0.22 },
  hoverlabel: { bgcolor: '#ffffff', bordercolor: '#d7dfd9', font: { color: '#20392e' } },
};

const CONFIG: Partial<Plotly.Config> = {
  displayModeBar: false,
  responsive: true,
};

interface Props {
  data: Plotly.Data[];
  layout?: Partial<Plotly.Layout>;
  height?: number;
}

export function Plot({ data, layout, height = 380 }: Props) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const merged: Partial<Plotly.Layout> = {
      ...PLOT_LAYOUT,
      ...layout,
      height,
      xaxis: { ...PLOT_LAYOUT.xaxis, ...layout?.xaxis },
      yaxis: { ...PLOT_LAYOUT.yaxis, ...layout?.yaxis },
    };
    Plotly.react(el, data, merged, CONFIG);
    return () => {
      Plotly.purge(el);
    };
  }, [data, layout, height]);

  return <div ref={ref} className="plot" />;
}
