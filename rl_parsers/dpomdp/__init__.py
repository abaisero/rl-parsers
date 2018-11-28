from ply import lex, yacc

from . import tokrules
from .parser import DPOMDP, Parser

__all__ = ['DPOMDP_Parser', 'parse']


class DPOMDP_Parser:
    def __init__(self, *, debug=False, optimize=True):
        self.debug = debug
        self.optimize = optimize
        self.lexer = lex.lex(module=tokrules, debug=debug, optimize=optimize)

    def parse_file(self, filename: str) -> DPOMDP:
        with open(filename) as f:
            return self.parse_string(f.read())

    def parse_string(self, string: str) -> DPOMDP:
        parser = yacc.yacc(
            module=Parser(), debug=self.debug, optimize=self.optimize
        )
        return parser.parse(string, lexer=self.lexer)


# legacy
def parse(string: str, **kwargs) -> DPOMDP:
    parser = DPOMDP_Parser(**kwargs)
    return parser.parse_string(string)
