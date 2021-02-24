// const ReactDOM = require('react-dom');
import ReactDOM from 'react-dom';
import './index.scss';
import React, { useState } from 'react';
import TeX from '@matejmazur/react-katex';
import { HashRouter, Link, Route, Switch, useParams } from 'react-router-dom';

interface Sample {
    latex: string;
    index: number;
}

function Term(): JSX.Element {
    const { index: index_str } = useParams<{ index: string }>();
    const index = parseInt(index_str);
    const [sample, changeSample] = useState<Sample | null>(null);

    if (sample !== null) {
        if (index != sample.index) {
            fetch(`./api/sample/${index}`)
                .then(response => response.json())
                .then(changeSample)
                .catch(console.error);
        }
        return (
            <div className="detail">
                <div className="term">
                    <TeX >{sample.latex}</TeX>
                </div>
                <div className="navbar">
                    <Link to={`/${index - 1}`}>Previous</Link>
                    <Link to={`/${index + 1}`}>Next</Link>
                </div>
            </div>
        );
    } else {
        fetch(`./api/sample/${index}`)
            .then(response => response.json())
            .then(changeSample)
            .catch(console.error);
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
                </Switch>
            </HashRouter>
        </div>
    );
}

ReactDOM.render(
    <Index />,
    document.getElementById('root')
);
