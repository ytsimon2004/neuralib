import textwrap

from neuralib.argp import AbstractParser
from neuralib.argp.dispatch import DispatchOption


class Test(AbstractParser, DispatchOption):
    target: str = DispatchOption.argument(
        '--run'
    )

    @classmethod
    def EPILOG(cls):
        return f"""\
Command (--run)
{textwrap.indent(Test.parser_command_epilog(), "  ")}
"""

    def run(self):
        self.invoke_command(self.target)

    @DispatchOption.dispatch('A')
    def run_a(self):
        """doc A."""
        print('a')

    @DispatchOption.dispatch('B')
    def run_b(self):
        """doc B"""
        print('b')


if __name__ == '__main__':
    Test().main()
