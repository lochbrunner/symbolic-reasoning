import torch


def predict_rule(model, term, ident_to_ix):
    # Get remainders of prev
    remainders = [predict_rule(model, child, ident_to_ix)[1] for child in term.childs]

    hidden = model.initial_hidden()

    ident_tensor = torch.tensor(
        [ident_to_ix[term.ident]], dtype=torch.long)

    if len(remainders) == 0:
        remainders = [model.initial_remainder()]

    for incoming_remainder in remainders:
        out, hidden, remainder = model(
            ident_tensor, incoming_remainder, hidden)

    # (out, hidden, remainder) = model(ident_tensor)

    return out, remainder


def create_batches(samples, batch_size):
    return [samples[i:i+batch_size] for i in range(0, len(samples), batch_size)]
