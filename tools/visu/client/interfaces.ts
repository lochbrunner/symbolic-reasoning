export interface Position {
    ruleId: number;
    path: number;
}

export interface Policy extends Position {
    policy: number;
}

export interface ErrorTops {
    tops: number[];
    total: number;
}

export type ValidationMetricNames = 'exact' |
    'exact-no-padding' |
    'when-rule' |
    'with-padding';

export interface ValidationMetrics {
    'exact': ErrorTops;
    'exact-no-padding': ErrorTops;
    'when-rule': ErrorTops;
    'with-padding': ErrorTops;
}

export interface Sample {
    latex: string;
    index: number;
    idents: string[];
    isOperator: boolean[];
    isFixed: boolean[];
    isNumber: boolean[];
    policy: Policy[]; // Deprecated
    indexMap: number[][];
    predictedValue: number;
    groundTruthValue: number;
    parts: string[];
    rules: string[];
    predictedPolicy: number[][];
    gtPolicy: number[][];
    fitMask: boolean[][];
    possibleFits: Position[];   // Deprecated
    validationMetrics: ValidationMetrics;
}