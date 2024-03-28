from __future__ import annotations

import attrs

__all__ = ['CliArgs']


@attrs.define
class CliArgs:
    flag: str | None = attrs.field(default=None)
    args: str | None = attrs.field(default=None)

    # noinspection PyUnresolvedReferences
    @flag.validator
    def _check_flag(self, attribute, value: str) -> None:
        if value is None:
            return

        if not isinstance(value, str):
            raise TypeError('')

        valid1 = value.startswith('-') and value[1:].isalpha()
        valid2 = value.startswith('--') and value[2].isalpha()
        if not (valid1 or valid2):
            raise ValueError(f'{self} {attribute.name} not valid')

    # noinspection PyUnresolvedReferences
    @args.validator
    def _check_args(self, attribute, value: str) -> None:
        if value is None:
            return

        if not isinstance(value, str):
            self.args = str(value)

    def as_command(self) -> list[str]:

        # position arg
        if self.flag is None and self.args is not None:
            return [self.args]

        # flag
        elif self.flag is not None and self.args is None:
            return [self.flag]

        # flag + arg
        elif self.flag is not None and self.args is not None:
            return [self.flag, self.args]
        else:
            raise RuntimeError('')
