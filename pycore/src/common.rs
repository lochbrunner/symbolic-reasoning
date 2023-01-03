use pyo3::class::basic::CompareOp;

pub fn op_to_string(op: &CompareOp) -> &str {
    match op {
        CompareOp::Lt => "<",
        CompareOp::Le => "<=",
        CompareOp::Eq => "==",
        CompareOp::Ne => "!=",
        CompareOp::Gt => ">",
        CompareOp::Ge => ">=",
    }
}
