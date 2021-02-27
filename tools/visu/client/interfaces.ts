export interface Sample {
    latex: string;
    index: number;
    idents: string[];
    isOperator: boolean[];
    isFixed: boolean[];
    isNumber: boolean[];
    policy: { ruleId: number, policy: number, path: number }[];
    indexMap: number[][];
    value: number;
    predictedValue: number;
    parts: string[];
    rules: string[];
    predictions: number[][];
    possibilities: { ruleId: number, path: number }[];
}