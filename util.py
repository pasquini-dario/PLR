END = '\n'
clean = lambda X: [x.decode().split(END)[0] for x in X]

def readP(path, encoding='iso-8859-1', n=0):
    """ read txt passwords file (\n sep) """
    with open(path, encoding=encoding) as f:
        raw = [x.strip() for x in f if x]
        if n:
            raw = [x for x in raw if len(x) <= n]
    return raw



def readP_skip_encoding (path, encoding='utf-8', MIN_LEN=0, MAX_LEN=0):
    print(f'MIN_LEN: {MIN_LEN}; MAX_LEN: {MAX_LEN}\nIT DOES REMOVE EMPTY STRINGS!')
    with open(path, encoding=encoding, errors='ignore') as f:
        raw = [x[:-1] for x in f]
        if MAX_LEN:
            X = [x for x in raw if x and len(x) <= MAX_LEN and len(x) >= MIN_LEN]
        else:
            X = [x for x in raw if x and len(x) >= MIN_LEN]
    return X

