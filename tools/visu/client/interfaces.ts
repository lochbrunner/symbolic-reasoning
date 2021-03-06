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
    policy: Policy[];
    indexMap: number[][];
    predictedValue: number;
    groundTruthValue: number;
    parts: string[];
    rules: string[];
    predictions: number[][];
    predictedPolicy: number[][];
    possibleFits: Position[];
    validationMetrics: ValidationMetrics;
}