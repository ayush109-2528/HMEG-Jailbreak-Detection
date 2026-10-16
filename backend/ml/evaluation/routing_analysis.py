from ml.ensemble.router import Router


def routers():
    return [Router(.1, .9), Router(.2, .8), Router(.3, .7), Router(.4, .6)]
