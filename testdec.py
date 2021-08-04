def conc_wrap(conc_func):
    def conc_wrapped(*args):
        try:
            conc_out = conc_func(*args)
        except KeyError:
            conc_out = 0.0
        return conc_out

    return conc_wrapped


@conc_wrap
def conc(ks, tot):
    return ks["abc"] * tot


tot = 3
ks1 = {"abc": 1.5}
ks2 = {"def": 1.5}

conc1 = conc(ks1, tot)
conc2 = conc(ks2, tot)
