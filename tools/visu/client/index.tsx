// const ReactDOM = require('react-dom');
import ReactDOM from 'react-dom';
import './index.scss';
import React from 'react';

function Index() {
    return <h1>Hello World!</h1>
}

ReactDOM.render(
    <Index />,
    document.getElementById('root')
);
