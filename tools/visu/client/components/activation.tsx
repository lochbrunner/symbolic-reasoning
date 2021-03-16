import TeX from '@matejmazur/react-katex';
import React, { useState } from 'react';
import Switch from '@material-ui/core/Switch';
declare function require(module: string): any;
const createColorMap = require('colormap');
import _ from 'lodash';

import './activation.scss'
import { FormControlLabel, FormGroup, FormLabel } from '@material-ui/core';
import { Position, Sample } from '../interfaces';
import Copy from '../components/copy';
import { copyToClipboard, createLatexTable, createLatexTableFiltered } from '../utils';

export interface Props {
    sample: Sample;
    sampleId: number;
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

function createMap(predictions: number[][], dx: number, dy: number): JSX.Element[] {
    const SHADES_COUNT = 100;
    const colors: string[] = createColorMap({
        colormap: 'jet',
        nshades: SHADES_COUNT,
        format: 'hex',
        alpha: 1
    });
    const min = Math.min(...predictions.map(s => Math.min(...s)));
    const max = Math.max(...predictions.map(s => Math.max(...s)));
    const color = (value: number) => {
        const normedValue = (value - min) * SHADES_COUNT / (max - min);
        return colors[Math.floor(normedValue)];
    };
    const width = `${dx}px`;
    const height = `${dy}px`;
    return predictions.map((path, iy) => path.map((rule, ix) => <rect key={`${ix}-${iy}`} y={`${(iy) * dy}px`} x={`${ix * dx}px`} width={width} height={height} style={{ fill: color(rule) }} ><title>{rule.toFixed(3)}</title></rect>)).flat();
}

function createMapFiltered(predictions: number[][], dx: number, dy: number, filter: Position[], ruleMap: { [orig: number]: number }): JSX.Element[] {
    const SHADES_COUNT = 100;
    const colors: string[] = createColorMap({
        colormap: 'jet',
        nshades: SHADES_COUNT,
        format: 'hex',
        alpha: 1
    });
    const apply = (ruleId: number, path: number) => {
        return filter.some(pos => pos.path === path && pos.ruleId === ruleId);
    }
    const min = Math.min(...predictions.map((paths, ruleId) => Math.min(...paths.filter((j, path) => apply(ruleId, path)))));
    const max = Math.max(...predictions.map((paths, ruleId) => Math.max(...paths.filter((j, path) => apply(ruleId, path)))));
    const color = (value: number) => {
        const normedValue = (value - min) * SHADES_COUNT / (max - min);
        return colors[Math.floor(normedValue)];
    };
    const width = `${dx}px`;
    const height = `${dy}px`;
    return predictions.map((paths, iy) => paths.map((rule, ix) => ({ rule, ix }))
        .filter((rule, ix) => apply(iy, ix))
        .map(({ rule, ix }) => <rect key={`${ix}-${iy}`} y={`${ruleMap[iy] * dy}px`} x={`${ix * dx}px`} width={width} height={height} style={{ fill: color(rule) }} ><title>{rule.toFixed(3)}</title></rect>)).flat();
}

function createSortedMap(predictions: number[][], dx: number, dy: number, filter: Position[], ruleMap: { [orig: number]: number }): JSX.Element[] {
    const colors: string[] = createColorMap({
        colormap: 'jet',
        nshades: filter.length,
        format: 'hex',
        alpha: 1
    });
    const possibilities = filter.map(position => ({ position, confidence: predictions[position.ruleId][position.path] })).sort((a, b) => a.confidence - b.confidence)
    const width = `${dx}px`;
    const height = `${dy}px`;
    return possibilities.map((possibility, i) => <g key={`pred-${i}`}>
        <rect y={`${ruleMap[possibility.position.ruleId] * dy}px`} x={`${possibility.position.path * dx}px`} width={width} height={height} style={{ fill: colors[i] }} ><title>{possibility.confidence}</title></rect>
        <text textAnchor="middle" dominantBaseline="middle" y={`${(ruleMap[possibility.position.ruleId] + 0.5) * dy}px`} x={`${(possibility.position.path + 0.5) * dx}px`}>{filter.length - i}</text>
    </g>);
}

function possibilitiesMap(possibilities: Position[], dx: number, dy: number): JSX.Element[] {
    const width = `${dx}px`;
    const height = `${dy}px`;
    return possibilities.map((possibility, i) => <rect key={`possibility-${i}`} className="possibility" y={`${(possibility.ruleId + 0) * dy}px`} x={`${possibility.path * dx}px`} height={height} width={width}></rect>);
}

export function render(props: Props) {
    const [size, changeSize] = useState<Size>({ width: 100, height: 100 });
    const [showGroundTruth, changeShowGroundTruth] = useState<boolean>(true);
    const [showIdents, changeShowIdents] = useState<boolean>(true);
    const [showOperator, changeShowOperator] = useState<boolean>(false);
    const [showNumber, changeShowNumber] = useState<boolean>(false);
    const [showFixed, changeShowFixed] = useState<boolean>(false);
    const [showIndexMap, changeShowIndexMap] = useState<boolean>(true);
    const [showPossibilities, changeShowPossibilities] = useState<boolean>(false);
    const [showPredictions, changeShowPredictions] = useState<boolean>(true);
    const [filterPossibilities, changeFilterPossibilities] = useState<boolean>(true);
    const [sortPossibilities, changeSortPossibilities] = useState<boolean>(true);

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

    const nx = sample.parts.length + 1;
    const dx = size.width / nx;
    const dy = 25;

    let layers: JSX.Element[] = [];
    // It would be possible to use a Hashset but dictionaries are less error prone.
    let ruleMap: { [orig: number]: number } | null = null;

    if (showPredictions) {
        if (filterPossibilities) {
            const usedRuleIds = new Set(sample.possibleFits.map(p => p.ruleId));
            ruleMap = _.fromPairs(Array.from(usedRuleIds.values()).map((p, i) => [p, i]));
            if (sortPossibilities) {
                layers = [...layers, ...createSortedMap(sample.predictedPolicy, dx, dy, sample.possibleFits, ruleMap)];
            } else {
                layers = [...layers, ...createMapFiltered(sample.predictedPolicy, dx, dy, sample.possibleFits, ruleMap)];
            }
        } else {
            layers = [...layers, ...createMap(sample.predictedPolicy, dx, dy)];
        }
    }
    const x_labels = sample.parts.map((l, i) => <div key={i} style={{
        right: `${(nx - i - 0.8) * dx}px`
    }}><TeX math={l} /></div>);
    let rules = sample.rules.map((rule, ruleId) => ({ rule, ruleId, origId: ruleId }));
    if (ruleMap !== null) {
        rules = rules.filter(({ rule, ruleId }) => ruleId in (ruleMap as any)).map(({ rule, ruleId }) => ({ rule, origId: ruleId, ruleId: (ruleMap as any)[ruleId] }));
    }
    const y_labels = rules.map(({ rule, ruleId }) => <div key={ruleId} style={{
        top: `${dy * ruleId}px`,
    }}><TeX math={rule} /></div>);

    if (showPossibilities && !(filterPossibilities && showPredictions)) {
        layers = [...layers, ...possibilitiesMap(sample.possibleFits, dx, dy)];
    }

    if (showGroundTruth) {
        const groundTruthValues = sample.policy.filter(gt => gt.ruleId > 0);
        let groundTruth;
        if (ruleMap !== null) {
            groundTruth = groundTruthValues.map((v, i) =>
                <rect key={`gt - ${i} `} className={`gt cell ${v.policy !== 1. ? 'negative' : 'positive'} `} x={v.path * dx} width={dx} y={(ruleMap as any)[v.ruleId] * dy} height={dy} />)
        } else {
            groundTruth = groundTruthValues.map((v, i) =>
                <rect key={`gt - ${i} `} className={`gt cell ${v.policy !== 1. ? 'negative' : 'positive'} `} x={v.path * dx} width={dx} y={v.ruleId * dy} height={dy} />)
        }
        layers = [...layers, ...groundTruth];
    }

    const download = () => {
        if (ruleMap !== null) {
            copyToClipboard(createLatexTableFiltered({ ruleMap, sample, rules, sampleId: props.sampleId }));
        }
        else {
            copyToClipboard(createLatexTable({ sample, rules, sampleId: props.sampleId }));
        }
    };

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

    const ny = y_labels.length;

    return (
        <div className="activation">
            <div className="main" >
                <div className="upper-container">
                    <div className="upper" style={{ height: `${ny * dy} px` }}>
                        <svg ref={updateSizeBind as any} preserveAspectRatio='none' style={{ height: `${ny * dy}px` }}>
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
                    <FormLabel component="legend">Output Layer</FormLabel>
                    <FormControlLabel label="Prediction" control={
                        <Switch checked={showPredictions} onChange={changeShowX(changeShowPredictions)} color="primary" />
                    } />
                    <FormControlLabel label="Possibilities" control={
                        <Switch checked={showPossibilities} disabled={showPredictions && (filterPossibilities || sortPossibilities)} onChange={changeShowX(changeShowPossibilities)} color="primary" />
                    } />
                    <FormControlLabel label="Filter Possibilities" control={
                        <Switch checked={filterPossibilities} disabled={!showPredictions} onChange={changeShowX(changeFilterPossibilities)} color="primary" />
                    } />
                    <FormControlLabel label="Sort Possibilities" control={
                        <Switch checked={sortPossibilities} disabled={!showPredictions || !filterPossibilities} onChange={changeShowX(changeSortPossibilities)} color="primary" />
                    } />
                    <FormControlLabel label="Ground Truth" control={
                        <Switch checked={showGroundTruth} onChange={changeShowX(changeShowGroundTruth)} color="primary" />
                    } />
                </FormGroup>
                <FormGroup row>
                    <FormLabel component="legend">Input Layer</FormLabel>
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
                    <div className="copy-table" onClick={e => download()}><Copy /></div>
                </FormGroup>
            </div>
        </div>
    );
}