import { lazy, Suspense } from 'react';

const PlotCanvas = lazy(() => import('./PlotCanvas').then(module => ({ default: module.Plot })));

export function Plot(props: { data: Plotly.Data[]; layout?: Partial<Plotly.Layout>; height?: number }) {
  return <Suspense fallback={<div className="empty" role="status" style={{ height: props.height ?? 380 }}><span className="spinner" /> Loading chart…</div>}><PlotCanvas {...props}/></Suspense>;
}
