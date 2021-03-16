import { Sample } from './interfaces';

export function copyToClipboard(text: string) {
    const dummy = document.createElement('textarea');
    document.body.appendChild(dummy);
    dummy.value = text;
    dummy.select();
    document.execCommand('copy');
    document.body.removeChild(dummy);
}

export function asDownload(text: string, filename: string) {
    const a = document.createElement('a');
    document.body.appendChild(a);
    a.style.display = 'none';
    const blob = new Blob([text], { type: 'octet/stream' });
    const url = window.URL.createObjectURL(blob);
    a.href = url;
    a.download = filename;
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
}

function fixColors(color: string) {
    const oc1 = '#aaaaaa';
    const oc2 = '#000077';
    const nc1 = 'Gray';
    const nc2 = 'MidnightBlue';
    return color.replace(oc1, nc1).replace(oc2, nc2);
}

export function createLatexTable(props: { sample: Sample, rules: { rule: string, ruleId: number, origId: number }[], sampleId: number }) {
    const { sampleId } = props;
    const rules = props.rules.map(r => `    $${r.rule}$`).join(',\n');
    const parts = props.sample.parts.map(p => `    $${fixColors(p)}$`).join(',\n');
    const activations = props.rules.map(rule => props.sample.predictions[rule.origId]);
    const max = Math.max(...activations.map(row => Math.max(...row.slice(0, -1))));
    const min = Math.min(...activations.map(row => Math.min(...row.slice(0, -1))));
    const scale = (v: number) => Math.ceil(((v - min) * 100 / (max - min)));

    const cells = activations.map(row => `    {${row.slice(0, -1).map(scale).join(',')}},`).join('\n');

    return `
% sample id: #${sampleId}
\\begin{tikzpicture}[scale=0.6]
\\foreach \\y [count=\\p] in {
${cells}
    } {
    \\foreach \\x [count=\\r] in \\y {
        \\node[fill=yellow!\\x!purple, minimum size=6mm, text=white] at (\\r,-\\p) {};
    }
}
\\foreach \\t [count=\\p] in {
${rules}
} {
    \\node[anchor=west] at (${1 + props.sample.parts.length}, -\\p) {\\t};
}
\\foreach \\t [count=\\r] in {
${parts}
} {
    \\node[anchor=west, rotate=-45] at (\\r - 0.2, ${(-0.7 - props.rules.length).toFixed(1)}) {\\t};
}
\\end{tikzpicture}
`;
}