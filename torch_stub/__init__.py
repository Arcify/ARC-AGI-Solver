class Tensor(list):
    pass

def tensor(data, dtype=None):
    return Tensor(data)

long = 'long'
