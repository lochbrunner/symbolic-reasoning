import React, { useState } from 'react';
import InfiniteScroll from 'react-infinite-scroll-component';
import TeX from '@matejmazur/react-katex';
import './overview.scss';
import { ValidationMetricNames } from '../interfaces';
import { useHistory, useParams } from 'react-router-dom';
import { TextField } from '@material-ui/core';

function encodeQueryData(data: any) {
    const ret = [];
    for (let d in data)
        ret.push(encodeURIComponent(d) + '=' + encodeURIComponent(data[d]));
    return ret.join('&');
}


interface Sample {
    initial: string;
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
    key: 'none' | 'name' | ValidationMetricNames | 'value-gt' | 'value-predicted' | 'value-error';
    up: boolean;
    filter: string;
}

export default function overview(): JSX.Element {
    const { sorting_key, direction, filter } = useParams<{ sorting_key: Sorting['key'], direction: string, filter: string }>();
    const sorting = { key: sorting_key, filter: filter ? decodeURIComponent(filter) : '', up: direction === 'up' };
    const [{ samples, prevSorting }, changeSamples] = useState<{ samples: Sample[], prevSorting: Sorting }>({ samples: [], prevSorting: sorting });
    const [size, changeSize] = useState<number | null>(null);
    const history = useHistory();
    const batch_size = 40;

    if (prevSorting.filter !== sorting.filter || prevSorting.key !== sorting.key || prevSorting.up !== sorting.up) {
        fetch(`./api/overview?${encodeQueryData({ begin: 0, end: batch_size, ...sorting })}`)
            .then(response => response.json())
            .then((samples: Sample[]) => changeSamples({ samples, prevSorting: sorting }))
            .catch(console.error);
    }

    console.log(sorting);

    const changeSorting = (newSorting: Sorting) => {
        const { filter, key, up } = newSorting;
        history.push(`/overview/${key}/${up ? 'up' : 'down'}/${encodeURIComponent(filter) ?? ''}`);
    };

    const updateSorting = (key: Sorting['key']) => {
        return () => {
            if (sorting_key === key)
                changeSorting({ ...sorting, up: !sorting.up });
            else
                changeSorting({ ...sorting, key, up: true });
        };
    };

    const changeFilter = (newFilter: string) => {
        changeSorting({ ...sorting, filter: newFilter });
    }

    const addSamples = (newSamples: Sample[]) => {
        changeSamples({ prevSorting, samples: [...samples, ...newSamples] });
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

    const positiveSpan = (value: boolean) => {
        if (value) {
            return <span className="positive">✓</span>;
        } else {
            return <span className="negative">✗</span>;
        }
    }

    if (size !== null) {
        const tops = (sample: Sample, label: ValidationMetricNames) => {
            const error = sample.summary[label];
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
                <header>
                    <h1>Overview</h1>
                    <div>
                        <TextField value={sorting.filter} onChange={e => changeFilter(e.target.value)} id="standard-search" label="filter by rule" type="search" />
                    </div>
                </header>
                <InfiniteScroll dataLength={samples.length} hasMore={size > samples.length} next={next} loader={<h1>Loading...</h1>}>
                    <table>
                        <thead>
                            <tr>
                                <th className="clickable" onClick={updateSorting('name')}>Initial</th>
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