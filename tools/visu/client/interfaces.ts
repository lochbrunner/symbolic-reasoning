export interface Position {
    ruleId: number;
    path: number;
}

export interface Policy extends Position {
    policy: number;
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
    validationMetrics: undefined;
}