import React, { useState } from 'react';
import InfiniteScroll from 'react-infinite-scroll-component';
import TeX from '@matejmazur/react-katex';
import './overview.scss';
import { ValidationMetricNames, ValidationMetrics } from '../interfaces';
import { useHistory } from 'react-router-dom';

function encodeQueryData(data: any) {
    const ret = [];
    for (let d in data)
        ret.push(encodeURIComponent(d) + '=' + encodeURIComponent(data[d]));
    return ret.join('&');
}


interface Sample {
    initial: string;
    validation: ValidationMetrics;
    index: number;
}

export default function overview(): JSX.Element {
    const [samples, changeSamples] = useState<Sample[]>([]);
    const [size, changeSize] = useState<number | null>(null);

    const addSamples = (newSamples: Sample[]) => {
        changeSamples([...samples, ...newSamples]);
    };

    const next = () => {
        if (overview.length < (size ?? 1)) {
            const begin = samples.length;
            const end = begin + 50;
            fetch(`./api/overview?${encodeQueryData({ begin, end })}`)
                .then(response => response.json())
                .then(addSamples)
                .catch(console.error);
        }

    };
    const history = useHistory();

    if (size !== null) {
        const tops = (sample: Sample, label: ValidationMetricNames) => {
            const error = sample.validation[label].tops.findIndex(v => v > 0);
            return <th>{error}</th>;
        };

        const rows: JSX.Element[] = samples.map((sample, i) => {
            return (
                <tr key={i} className="row" onClick={() => history.push(`/detail/${sample.index}`)}>
                    <th><TeX>{sample.initial}</TeX></th>
                    {tops(sample, 'exact-no-padding')}
                    {tops(sample, 'exact')}
                    {tops(sample, 'when-rule')}
                    {tops(sample, 'with-padding')}
                </tr>);
        });
        return (
            <div className="overview">
                <h1>Overview</h1>
                <InfiniteScroll dataLength={samples.length} hasMore={size > samples.length} next={next} loader={<h1>Loading...</h1>}>
                    <table>
                        <thead>
                            <tr>
                                <th>Initial</th>
                                <th>exact no padding</th>
                                <th>exact</th>
                                <th>when rule</th>
                                <th>with padding</th>
                            </tr>
                        </thead>
                        <tbody>
                            {rows}
                        </tbody>
                    </table>
                </InfiniteScroll>
            </div>
        );
    }
    else {
        fetch('/api/length')
            .then(response => response.json())
            .then(response => changeSize(response.length))
            .catch(console.error);
        next();
        return <h1>Loading ...</h1>
    }
}