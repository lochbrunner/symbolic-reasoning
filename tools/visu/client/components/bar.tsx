import React from 'react';
import './bar.scss';
import { Tooltip } from '@material-ui/core';

export default function (props: { positive: number, negative: number, na: number }) {
    const { positive, negative, na } = props;
    const total = positive + negative + na;

    const style = (value: number) => ({
        width: `${value * 100 / total}px`
    });
    return (
        <Tooltip title={`${positive}/${negative}/${na}`}>
            <div className="bar" >
                <div className="positive" style={style(positive)} />
                <div className="negative" style={style(negative)}></div>
                <div className="na" style={style(na)}></div>
            </div>
        </Tooltip>
    );
}