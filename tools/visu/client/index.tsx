import ReactDOM from 'react-dom';
import './index.scss';
import React, { useState } from 'react';
import TeX from '@matejmazur/react-katex';
import { HashRouter, Link, Redirect, Route, Switch, useHistory, useParams } from 'react-router-dom';

import { render as Activation } from './components/activation';
import { Sample, ValidationMetrics, ErrorTops } from './interfaces';
import { Tooltip } from '@material-ui/core';


function Value(props: { gt: number, predicted: number }): JSX.Element {
    let mark;
    if (props.gt > 0.5) {
        mark = <span className="contributed">✓</span>;
    } else {
        mark = <span className="not-contributed">✗</span>;
    }
    return (
        <div className="value info">
            <span className="number">{props.predicted.toFixed(2)}</span>
            <span className="mark">{mark}</span>
        </div>
    );
}

function TopBar(props: { tops: ErrorTops }) {
    const error = props.tops.tops.findIndex(v => v > 0);
    if (error == -1) {
        return <div className="failed">-</div>;
    } else {
        const heightNumber = Math.max(15, (10 - error) * 10);
        const height = `${heightNumber}%`
        return <div style={{ height }} className="succeeded">{error}</div>;
    }
}

function Tops(props: { tops: ErrorTops, label: string }) {
    return (
        <Tooltip title={<span style={{ fontSize: '120%' }}>{props.label}</span>}>
            <div className="tops">
                <TopBar tops={props.tops} />
            </div >
        </Tooltip>
    );
}

function Validation(props: { validationMetrics: ValidationMetrics }): JSX.Element {
    return (
        <div className="validation info">
            <Tops tops={props.validationMetrics['exact-no-padding']} label="exact no padding" />
            <Tops tops={props.validationMetrics['exact']} label="exact with padding" />
            <Tops tops={props.validationMetrics['when-rule']} label="when rule" />
            <Tops tops={props.validationMetrics['with-padding']} label="when rule with padding" />
        </div>
    );
}

function Term(): JSX.Element {
    const { index: index_str } = useParams<{ index: string }>();
    const history = useHistory();
    const index = parseInt(index_str);
    const next = `/${index + 1}`;
    const prev = `/${index - 1}`;
    const [sample, changeSample] = useState<Sample | null>(null);

    const onKeyDown = (e: KeyboardEvent) => {
        if (e.key === 'ArrowLeft') {
            history.push(prev);
        } else if (e.key === 'ArrowRight') {
            history.push(next);
        } else {
            console.info(e.key);
        }
    };

    const newSample = () => {
        fetch(`./api/sample/${index}`)
            .then(response => response.json())
            .then(changeSample)
            .catch(console.error);
    };

    if (sample !== null) {
        if (index != sample.index) {
            newSample();
        }
        return (
            <div className="detail">
                <div className="summary">
                    <div className="term">
                        <TeX >{sample.latex}</TeX>
                    </div>
                    <Value gt={sample.groundTruthValue} predicted={sample.predictedValue} />
                    <Validation validationMetrics={sample.validationMetrics} />
                </div >
                <Activation sample={sample} />
                <div onKeyDown={onKeyDown as any} className="navbar">
                    <Link to={next}>Previous</Link>
                    <Link to={prev}>Next</Link>
                </div>
            </div >
        );
    } else {
        newSample();
        return <div>Empty</div>;
    }
}

function Index() {
    return (
        <div>
            <HashRouter>
                <Switch>
                    <Route path='/:index' >
                        <Term key={location.hash} />
                    </Route>
                    <Route path='/'>
                        <Redirect to="/1" />
                    </Route>
                </Switch>
            </HashRouter>
        </div>
    );
}

ReactDOM.render(
    <Index />,
    document.getElementById('root')
);
