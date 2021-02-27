export interface Sample {
    latex: string;
    index: number;
    idents: string[];
    isOperator: boolean[];
    isFixed: boolean[];
    isNumber: boolean[];
    policy: { ruleId: number, policy: number, path: number }[];
    indexMap: number[][];
    predictedValue: number;
    groundTruthValue: number;
    parts: string[];
    rules: string[];
    predictions: number[][];
    predictedPolicy: number[][];
}