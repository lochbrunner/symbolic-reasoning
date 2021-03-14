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
    value: {
        gt: boolean;
        predicted: number;
        error: number;
    }
    policy_gt: {
        positive: number;
        negative: number;
    };

    summary: {
        'exact': number;
        'exact-no-padding': number;
        'when-rule': number;
        'with-padding': number;
    }
}

interface Sorting {
    key: 'none' | ValidationMetricNames | 'value-gt' | 'value-predicted' | 'value-error';
    up: boolean;
}

export default function overview(): JSX.Element {
    const [samples, changeSamples] = useState<Sample[]>([]);
    const [size, changeSize] = useState<number | null>(null);
    const [sorting, changeSorting] = useState<Sorting>({ key: 'none', up: true, });

    const batch_size = 40;

    function reset(newSorting: Sorting) {
        fetch(`./api/overview?${encodeQueryData({ begin: 0, end: batch_size, ...sorting })}`)
            .then(response => response.json())
            .then(changeSamples)
            .catch(console.error);
        changeSorting(newSorting);
    }

    const updateSorting = (key: Sorting['key']) => {
        return () => {
            if (sorting.key === key)
                reset({ ...sorting, up: !sorting.up });
            else
                reset({ key, up: true });
        };
    };

    const addSamples = (newSamples: Sample[]) => {
        changeSamples([...samples, ...newSamples]);
    };

    const next = () => {
        if (overview.length < (size ?? 1)) {
            const begin = samples.length;
            const end = begin + batch_size;
            fetch(`./api/overview?${encodeQueryData({ begin, end, ...sorting })}`)
                .then(response => response.json())
                .then(addSamples)
                .catch(console.error);
        }

    };
    const history = useHistory();
    const positiveSpan = (value: boolean) => {
        if (value) {
            return <span className="positive">✓</span>;
        } else {
            return <span className="negative">✗</span>;
        }
    }

    if (size !== null) {
        const tops = (sample: Sample, label: ValidationMetricNames) => {
            const error = sample.validation[label].tops.findIndex(v => v > 0);
            return <td>{error}</td>;
        };

        const rows: JSX.Element[] = samples.map((sample, i) => {
            const { positive, negative } = sample.policy_gt;
            return (
                <tr key={i} className="row" onClick={() => history.push(`/detail/${sample.index}`)}>
                    <td><TeX>{sample.initial}</TeX></td>
                    {tops(sample, 'exact-no-padding')}
                    {tops(sample, 'exact')}
                    {tops(sample, 'when-rule')}
                    {tops(sample, 'with-padding')}
                    <td>{positiveSpan(sample.value.gt)}</td>
                    <td>
                        <span>{sample.value.error.toFixed(2)}</span>
                        <span>({sample.value.predicted.toFixed(2)})</span>
                    </td>
                    <td>{positive > 0 ? <span className="positive">{`${positive} ✓`}</span> : ''}
                        {negative > 0 ? <span className="negative">{`${negative} ✗`}</span> : ''}
                    </td>
                </tr>);
        });
        return (
            <div className="overview">
                <h1>Overview</h1>
                <InfiniteScroll dataLength={samples.length} hasMore={size > samples.length} next={next} loader={<h1>Loading...</h1>}>
                    <table>
                        <thead>
                            <tr>
                                <th className="clickable" onClick={updateSorting('none')}>Initial</th>
                                <th className="clickable" onClick={updateSorting('exact-no-padding')}>exact no padding</th>
                                <th className="clickable" onClick={updateSorting('exact')}>exact</th>
                                <th className="clickable" onClick={updateSorting('when-rule')}>when rule</th>
                                <th className="clickable" onClick={updateSorting('with-padding')}>with padding</th>
                                <th className="clickable" onClick={updateSorting('value-gt')}>contributed</th>
                                <th className="clickable" onClick={updateSorting('value-error')}>value error (predicted)</th>
                                <th >gt rules</th>
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