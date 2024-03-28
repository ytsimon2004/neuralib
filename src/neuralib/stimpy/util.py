import builtins


def try_casting_number(value: str, eval=False):
    """ try casting of potential number text to a number value"""
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            if eval:
                try:
                    return builtins.eval(value)
                except (SyntaxError, NameError):
                    pass

    return value


def unfold_stimuli_condition(parts: list[str]) -> list[list]:
    """unfold the numbers of stimuli (each row in prot file) for parsing

    .. seealso::

        :func:`stimpy._utils.dataframe._generate_extended_dataframe`

    """
    ret = []
    if '-' in parts[0]:
        nstim = list(map(int, parts[0].split('-', maxsplit=1)))
        nstim = tuple((nstim[0], nstim[1] + 1))
        ranging = nstim[1] - nstim[0]  # how many n in each row

        for n in range(*nstim):
            parts[0] = f'{n}'
            ext = []
            for i, it in enumerate(parts):
                if '{i}' in it:
                    it = it.replace('{i}', f'{n - 1}%{ranging}')
                if '{t}' in it:
                    raise NotImplementedError('')

                ext.append(try_casting_number(it, eval=True))

            ret.append(ext)

    else:
        for i, it in enumerate(parts):
            ret.append(try_casting_number(it, eval=True))

        ret = [ret]
    return ret
