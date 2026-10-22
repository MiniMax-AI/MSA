"""Host-only selection of the TP4 eight-token MegaMoE backend."""


def select_backend(backend, q_len):
    if backend != "mega":
        raise ValueError("backend must be 'mega'")
    if type(q_len) is not int or q_len != 8:
        raise ValueError("the mega backend supports only q_len=8")
    return "mega"
