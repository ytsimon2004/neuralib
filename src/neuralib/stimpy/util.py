from __future__ import annotations

__all__ = ['try_casting_number',
           'unfold_stimuli_condition']


def try_casting_number(value: str, do_eval: bool = False) -> float | int | str:
    """
    try casting of string to numerical value

    :param value: str
    :param do_eval: do built in eval()
    :return:
    """
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            if do_eval:
                try:
                    return eval(value)
                except (SyntaxError, NameError):
                    pass

    return value


def unfold_stimuli_condition(parts: list[str]) -> list[list]:
    """
    unfold the numbers of stimuli (each row in prot file) for parsing

    :param parts: list of str foreach row in the visual stimuli dataframe

        e.g., ['1', '3', '0', '0', '1', '0.04', '0', '0', '200', '200', "{'phase':['linear',1]}"]
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

                ext.append(try_casting_number(it, do_eval=True))

            ret.append(ext)

    else:
        for i, it in enumerate(parts):
            ret.append(try_casting_number(it, do_eval=True))

        ret = [ret]
    return ret
