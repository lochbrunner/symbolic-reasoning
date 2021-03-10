import ReactDOM from 'react-dom';
import React from 'react';
import './index.scss';
import { HashRouter, Route, Switch } from 'react-router-dom';
import Detail from './pages/detail';
import Overview from './pages/overview';


function Index() {
    return (
        <div>
            <HashRouter>
                <Switch>
                    <Route path='/detail/:index' >
                        <Detail key={location.hash} />
                    </Route>
                    <Route path='/'>
                        <Overview />
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
