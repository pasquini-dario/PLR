END = '\n'
clean = lambda X: [x.decode().split(END)[0] for x in X]