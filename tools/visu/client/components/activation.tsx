import TeX from '@matejmazur/react-katex';
import React, { useState } from 'react';
import Switch from '@material-ui/core/Switch';

import './activation.scss'
import { FormControlLabel, FormGroup } from '@material-ui/core';
import { Sample } from '../interfaces';

export interface Props {
    sample: Sample;
}

interface Size {
    width: number;
    height: number;
}

function IdentLayer(props: { dx: number, idents: string[] }): JSX.Element {
    return (
        <div className="row">
            <div className="label">Idents</div>
            <div className="content">{
                props.idents.map((ident, i) => <div key={i} style={{
                    left: `${props.dx * i}px`, width: `${props.dx}px`

                }}> <span>{ident}</span></div>)
            }</div>
        </div >
    );
}

function IndexMap(props: { dx: number, indexMap: number[][] }): JSX.Element {
    const kernel = ['self', 'child-1', 'child-2', 'parent'];
    const [hovered, changeHovered] = useState<number | null>(null);

    const enter = (index: number) => (element: any) => {
        changeHovered(index);
    };

    const leave = (index: number) => (element: any) => {
        changeHovered(null);
    };

    const bgMap: { [key: number]: string } = {};

    if (hovered !== null) {
        bgMap[props.indexMap[hovered][0]] = `${kernel[0]}-bg`;
        bgMap[props.indexMap[hovered][1]] = `${kernel[1]}-bg`;
        bgMap[props.indexMap[hovered][2]] = `${kernel[2]}-bg`;
        bgMap[props.indexMap[hovered][3]] = `${kernel[3]}-bg`;
    }

    return (
        <div className="row">
            <div className="label">Index Map</div>
            <div className="content index-map">{
                props.indexMap.map((indices, i) => <div key={i} className={bgMap[i] || ''} onMouseEnter={enter(i)} onMouseLeave={leave(i)}
                    style={{
                        left: `${props.dx * i}px`, width: `${props.dx}px`

                    }}>{indices.map((index, j) => <span key={j} className={kernel[j]}>{index}</span>)}</div>)
            }</div>
        </div >
    );
}

function BooleanLayer(props: { dx: number, values: boolean[], label: string }): JSX.Element {
    const content = (value: boolean) => {
        if (value) {
            return <span className="success">✓</span>;
        } else {
            return <span className="failed">✗</span>;
        }
    };
    return (
        <div className="row">
            <div className="label">{props.label}</div>
            <div className="content boolean">{
                props.values.map((value, i) => <div key={i} style={{
                    left: `${props.dx * i}px`, width: `${props.dx}px`
                }}> {content(value)}</div>)
            }</div>
        </div >
    );
}

export function render(props: Props) {
    const [size, changeSize] = useState<Size>({ width: 100, height: 100 });
    const [showGroundTruth, changeShowGroundTruth] = useState<boolean>(true);
    const [showIdents, changeShowIdents] = useState<boolean>(true);
    const [showOperator, changeShowOperator] = useState<boolean>(false);
    const [showNumber, changeShowNumber] = useState<boolean>(false);
    const [showFixed, changeShowFixed] = useState<boolean>(false);
    const [showIndexMap, changeShowIndexMap] = useState<boolean>(true);

    const { sample } = props;

    const changeShowX = (callback: (value: boolean) => void) => (event: React.ChangeEvent<any>) => {
        callback(event.target.checked);
    }

    const updateSizeBind = (element: SVGAElement | null) => {
        if (!element) return;
        const { clientWidth, clientHeight } = element;

        if (size.height !== clientHeight || size.width !== clientWidth) {
            changeSize({ width: clientWidth, height: clientHeight })
        }
    };

    const ny = sample.rules.length;
    const nx = sample.parts.length + 1;
    const dx = size.width / nx;
    const dy = 25;

    const x_labels = sample.parts.map((l, i) => <div key={i} style={{
        right: `${(nx - i - 0.8) * dx}px`
    }}><TeX math={l} /></div>);
    const y_labels = sample.rules.map((l, i) => <div key={i} style={{
        top: `${dy * i}px`,
    }}><TeX math={l} /></div>);

    let layers: JSX.Element[] = [];

    if (showGroundTruth) {
        const groundTruthValues = sample.policy.filter(gt => gt.ruleId > 0);
        const groundTruth = groundTruthValues.map((v, i) =>
            <rect key={`gt-${i}`} className={`gt cell ${v.policy !== 1. ? 'negative' : 'positive'}`} x={v.path * dx} width={dx} y={v.ruleId * dy} height={dy} />)
        layers = [...layers, ...groundTruth];
    }

    let inputLayers: JSX.Element[] = [];

    if (showIdents) {
        inputLayers = [...inputLayers, <IdentLayer key="ident" dx={dx} idents={sample.idents} />];
    }
    if (showIndexMap) {
        inputLayers = [...inputLayers, <IndexMap key="index-map" dx={dx} indexMap={sample.indexMap} />];
    }
    if (showOperator) {
        inputLayers = [...inputLayers, <BooleanLayer key="operator" label="operator" dx={dx} values={sample.isOperator} />]
    }
    if (showFixed) {
        inputLayers = [...inputLayers, <BooleanLayer key="fixed" label="fixed" dx={dx} values={sample.isFixed} />]
    }
    if (showNumber) {
        inputLayers = [...inputLayers, <BooleanLayer key="number" label="number" dx={dx} values={sample.isNumber} />]
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
                <div className="input-layers">
                    {inputLayers}
                </div>
                <div className="lower">
                    <div className="x-label">{x_labels}</div>
                </div>
            </div>
            <div className="layer-control">
                <FormGroup row>
                    <FormControlLabel label="Ground Truth" control={
                        <Switch checked={showGroundTruth} onChange={changeShowX(changeShowGroundTruth)} color="primary" />
                    } />
                    <FormControlLabel label="Idents" control={
                        <Switch checked={showIdents} onChange={changeShowX(changeShowIdents)} color="primary" />
                    } />
                    <FormControlLabel label="Index Map" control={
                        <Switch checked={showIndexMap} onChange={changeShowX(changeShowIndexMap)} color="primary" />
                    } />
                    <FormControlLabel label="Operator" control={
                        <Switch checked={showOperator} onChange={changeShowX(changeShowOperator)} color="primary" />
                    } />
                    <FormControlLabel label="Fixed" control={
                        <Switch checked={showFixed} onChange={changeShowX(changeShowFixed)} color="primary" />
                    } />
                    <FormControlLabel label="Number" control={
                        <Switch checked={showNumber} onChange={changeShowX(changeShowNumber)} color="primary" />
                    } />
                </FormGroup>
            </div>
        </div>
    );
}