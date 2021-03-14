import ReactDOM from 'react-dom';
import React from 'react';
import './index.scss';
import { HashRouter, Redirect, Route, Switch } from 'react-router-dom';
import Detail from './pages/detail';
import Overview from './pages/overview';
import Histogram from './pages/histogram';


function Index() {
    return (
        <div>
            <HashRouter>
                <Switch>
                    <Route path='/detail/:index' >
                        <Detail key={location.hash} />
                    </Route>
                    <Route path='/histogram' >
                        <Histogram />
                    </Route>
                    <Route path='/overview/:sorting_key/:direction/:filter?'>
                        <Overview key={location.hash} />
                    </Route>
                    <Route path='/'>
                        <Redirect to='/overview/none/up/' />
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
