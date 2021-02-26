// const ReactDOM = require('react-dom');
import ReactDOM from 'react-dom';
import './index.scss';
import React, { useState } from 'react';
import TeX from '@matejmazur/react-katex';
import { HashRouter, Link, Redirect, Route, Switch, useHistory, useParams } from 'react-router-dom';

import { render as Activation } from './components/activation';

interface Sample {
    latex: string;
    index: number;
    x: number[][];
    policy: { ruleId: number, policy: number, path: number }[];
    s: number[][];
    v: number[];
    parts: string[];
    rules: string[];
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
        const groundTruth = sample.policy.filter(gt => gt.ruleId > 0);

        return (
            <div className="detail">
                <div className="term">
                    <TeX >{sample.latex}</TeX>
                </div>
                <Activation xLabels={sample.parts} yLabels={sample.rules} groundTruth={groundTruth} values={sample.x} />
                <div onKeyDown={onKeyDown as any} className="navbar">
                    <Link to={next}>Previous</Link>
                    <Link to={prev}>Next</Link>
                </div>
            </div>
        );
    } else {
        newSample();
        return <div>Empty</div>;
    }
}

function Index() {
    return (
        <div>
            <header>
                <h1>Live Visu</h1>
            </header>
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
