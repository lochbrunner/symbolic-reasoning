import React, { useState } from 'react';
import Plot from 'react-plotly.js';

import './histogram.scss';

interface Histogram {
    bin_edges: number[];
    hist: number[];
}

interface Histograms {
    exact: Histogram;
    exact_no_padding: Histogram;
    when_rule: Histogram;
    with_padding: Histogram;
}

function Content(): JSX.Element {
    const [histogram, changeHistogram] = useState<Histograms | null>(null);

    if (histogram !== null) {
        return (
            <div>
                <Plot
                    data={[{
                        x: histogram.exact_no_padding.bin_edges as any,
                        y: histogram.exact_no_padding.hist as any,
                        type: 'scatter',
                        name: 'exact no padding'
                    },
                    {
                        x: histogram.exact.bin_edges as any,
                        y: histogram.exact.hist as any,
                        type: 'scatter',
                        name: 'exact'
                    },
                    {
                        x: histogram.when_rule.bin_edges as any,
                        y: histogram.when_rule.hist as any,
                        type: 'scatter',
                        name: 'when rule'
                    },
                    {
                        x: histogram.with_padding.bin_edges as any,
                        y: histogram.with_padding.hist as any,
                        type: 'scatter',
                        name: 'with padding'
                    },
                    ]}
                    layout={{ width: 1200, height: 400, title: 'histogram' }}
                />
            </div>
        );
    } else {
        fetch('./api/histogram')
            .then(response => response.json())
            .then(changeHistogram)
            .catch(console.error);
        return <h1>Loading</h1>
    }
}

export default function render(): JSX.Element {


    return (
        <div className="histogram">
            <header>

            </header>
            <h1>Histogram</h1>
            <Content />
        </div>
    );
}