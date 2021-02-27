import TeX from '@matejmazur/react-katex';
import React, { useState } from 'react';
import Switch from '@material-ui/core/Switch';

import './activation.scss'
import { FormControlLabel, FormGroup } from '@material-ui/core';

export interface Props {
    xLabels: string[];
    yLabels: string[];
    values: number[][];
    groundTruth: { ruleId: number, path: number, policy: number }[];
}

interface Size {
    width: number;
    height: number;
}

export function render(props: Props) {
    const [size, changeSize] = useState<Size>({ width: 100, height: 100 });
    const [showGroundTruth, changeShowGroundTruthRaw] = useState<boolean>(true);

    const changeShowGroundTruth = (event: React.ChangeEvent<any>) => {
        changeShowGroundTruthRaw(event.target.checked);
    }

    const updateSizeBind = (element: SVGAElement | null) => {
        console.info(element);
        if (!element) return;
        const { clientWidth, clientHeight } = element;
        console.info(`clientWidth: ${clientWidth}`);
        console.info(`clientHeight: ${clientHeight}`);

        if (size.height !== clientHeight || size.width !== clientWidth) {
            changeSize({ width: clientWidth, height: clientHeight })
        }
    };

    const ny = props.yLabels.length;
    const nx = props.xLabels.length;
    const dx = size.width / nx;
    const dy = 25;

    const x_labels = props.xLabels.map((l, i) => <div key={i} style={{
        right: `${(nx - i - 0.8) * dx}px`
    }}><TeX math={l} /></div>);
    const y_labels = props.yLabels.map((l, i) => <div key={i} style={{
        top: `${dy * i}px`,
    }}><TeX math={l} /></div>);

    let layers: JSX.Element[] = [];

    if (showGroundTruth) {
        const groundTruth = props.groundTruth.map((v, i) =>
            <rect key={`gt-${i}`} className={`gt cell ${v.policy !== 1. ? 'negative' : 'positive'}`} x={v.path * dx} width={dx} y={v.ruleId * dy} height={dy} />)
        layers = [...layers, ...groundTruth];
    }

    return (
        <div className="activation">

            <div className="main" >
                <div className="upper-container">
                    <div className="upper" style={{ height: `${ny * dy}px` }}>
                        <svg ref={updateSizeBind as any} preserveAspectRatio='none' height={`${ny * 18}px`}>
                            {layers}
                        </svg>
                        <div className="y-label">{y_labels}</div>
                    </div>
                </div>
                <div className="lower">
                    <div className="x-label">{x_labels}</div>
                </div>
            </div>
            <div className="layer-control">
                <FormGroup row>
                    <FormControlLabel label="Ground Truth" control={
                        <Switch checked={showGroundTruth} onChange={changeShowGroundTruth} color="primary" />
                    } />
                </FormGroup>
            </div>
        </div>
    );
}