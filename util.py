END = '\n'
clean = lambda X: [x.decode().split(END)[0] for x in X]

def readP(path, encoding='iso-8859-1', n=0):
    """ read txt passwords file (\n sep) """
    with open(path, encoding=encoding) as f:
        raw = [x.strip() for x in f if x]
        if n:
            raw = [x for x in raw if len(x) <= n]
    return raw