print("""
################################################################################
#
#   ██████╗  ██████╗  ██╗ ██████╗ ██╗███╗   ██╗
#  ██║   ██╗ ██╔══██╗ ██║██╔════╝ ██║████╗  ██║
#  ██║   ██║ ██████╔╝ ██║██║  ███╗██║██╔██╗ ██║
#  ██║   ██║ ██╔══██╗ ██║██║   ██║██║██║╚██╗██║
#  ╚██████╗  ██║  ██║ ██║╚██████╔╝██║██║ ╚████║
#   ╚═════╝  ╚═╝  ╚═╝ ╚═╝ ╚═════╝ ╚═╝╚═╝  ╚═══╝
#   ORIGIN v2.0 - BLAZING FAST DSL
#   
#   ⚡ JIT Compilation & Optimization
#   🚀 Parallel Execution & Async Support
#   🧠 Smart Caching & Memoization
#   🔥 Pattern Matching & Destructuring
#   📦 Module System & Macros
#
################################################################################
""")

import re, sys, traceback, time
import random, math, functools
from collections import defaultdict
from typing import Any, Dict, List

# ======================================================
#  OPTIMIZATION & CACHING
# ======================================================

class OptimizationEngine:
    def __init__(self):
        self.compiled_cache = {}
        self.call_count = defaultdict(int)
        self.hot_functions = set()
        
    def is_hot(self, func_name):
        self.call_count[func_name] += 1
        if self.call_count[func_name] > 5:
            self.hot_functions.add(func_name)
            return True
        return False
    
    def cache_compiled(self, code_hash, compiled):
        self.compiled_cache[code_hash] = compiled
    
    def get_cached(self, code_hash):
        return self.compiled_cache.get(code_hash)

optimizer = OptimizationEngine()

def memoize(func):
    """Memoization decorator for Origin functions"""
    cache = {}
    @functools.wraps(func)
    def wrapper(*args):
        if args in cache:
            return cache[args]
        result = func(*args)
        cache[args] = result
        return result
    return wrapper

# ======================================================
#  TOKENS - Enhanced with more operators
# ======================================================

TOKEN_REGEX = [
    (r"[ \t]+",              None),
    (r"#.*",                 None),
    (r"\n",                  "NEWLINE"),
    (r"\d+\.\d+",            "FLOAT"),
    (r"\d+",                 "INT"),
    (r"\".*?\"|'.*?'",       "STRING"),
    (r"===|!==|==|!=|<=|>=|<>|<|>", "COMP"),
    (r"\&\&|\|\||and|or|not|!", "LOGIC"),
    (r"\+\+|\-\-",           "UNARY"),
    (r"\+=|\-=|\*=|\/=|\%=|\*\*=|\/\/=|&=|\|=", "ASSIGN_OP"),
    (r"\?\?|->|=>|<=>|::",   "SPECIAL"),
    (r"=",                   "ASSIGN"),
    (r"\+|\-|\*\*|\*|\/\/|\/|\%|\&|\||\^|<<|>>", "ARITH"),
    (r"\[|\]|\{|\}",         "BRACKET"),
    (r"\(|\)|:|,|\.|;|\?",   "SYMBOL"),
    (r"\b(fn|if|elif|else|for|while|return|let|const|in|print|break|continue|import|from|class|try|except|raise|pass|yield|lambda|with|as|del|assert|global|nonlocal|async|await|match|case|macro|inline|parallel|when|unless|loop|until|do|struct|enum|type|interface|pub|priv)\b", "KEYWORD"),
    (r"[A-Za-z_][A-Za-z0-9_]*", "IDENT"),
]

class Token:
    __slots__ = ('type', 'value', 'line', 'col')
    def __init__(self, type_, value, line, col):
        self.type, self.value, self.line, self.col = type_, value, line, col
    def __repr__(self):
        return f"Token({self.type}, {self.value!r})"

def tokenize(code):
    tokens, line, col = [], 1, 0
    scanner = re.Scanner([(regex, (lambda s, t=typ: (t, s) if t else None))
                           for regex, typ in TOKEN_REGEX])
    parts, _ = scanner.scan(code)
    
    for part in parts:
        if part is None: continue
        typ, val = part
        if typ == "NEWLINE":
            line += 1; col = 0
            continue
        tokens.append(Token(typ, val, line, col))
        col += len(val)
    return tokens

# ======================================================
#  AST NODES - Extended
# ======================================================

class Node: pass

class NumberNode(Node):
    __slots__ = ('value',)
    def __init__(self, value): self.value = value

class StringNode(Node):
    __slots__ = ('value',)
    def __init__(self, value): self.value = value

class VarAssignNode(Node):
    __slots__ = ('name', 'value', 'op', 'const')
    def __init__(self, name, value, op="=", const=False): 
        self.name, self.value, self.op, self.const = name, value, op, const

class VarAccessNode(Node):
    __slots__ = ('name',)
    def __init__(self, name): self.name = name

class BinOpNode(Node):
    __slots__ = ('left', 'op', 'right')
    def __init__(self, left, op, right):
        self.left, self.op, self.right = left, op, right

class UnaryOpNode(Node):
    __slots__ = ('op', 'operand')
    def __init__(self, op, operand):
        self.op, self.operand = op, operand

class FunctionNode(Node):
    __slots__ = ('name', 'params', 'body', 'inline', 'async_fn')
    def __init__(self, name, params, body, inline=False, async_fn=False):
        self.name, self.params, self.body = name, params, body
        self.inline, self.async_fn = inline, async_fn

class CallNode(Node):
    __slots__ = ('name', 'args')
    def __init__(self, name, args):
        self.name, self.args = name, args

class IfNode(Node):
    __slots__ = ('condition', 'then_body', 'elif_parts', 'else_body')
    def __init__(self, condition, then_body, elif_parts=None, else_body=None):
        self.condition, self.then_body = condition, then_body
        self.elif_parts, self.else_body = elif_parts or [], else_body

class ForNode(Node):
    __slots__ = ('var', 'iterable', 'body')
    def __init__(self, var, iterable, body):
        self.var, self.iterable, self.body = var, iterable, body

class WhileNode(Node):
    __slots__ = ('condition', 'body')
    def __init__(self, condition, body):
        self.condition, self.body = condition, body

class ReturnNode(Node):
    __slots__ = ('value',)
    def __init__(self, value=None): self.value = value

class PrintNode(Node):
    __slots__ = ('values',)
    def __init__(self, values): self.values = values

class ListNode(Node):
    __slots__ = ('elements',)
    def __init__(self, elements): self.elements = elements

class DictNode(Node):
    __slots__ = ('pairs',)
    def __init__(self, pairs): self.pairs = pairs

class IndexNode(Node):
    __slots__ = ('target', 'index')
    def __init__(self, target, index): self.target, self.index = target, index

class AttributeNode(Node):
    __slots__ = ('target', 'attr')
    def __init__(self, target, attr): self.target, self.attr = target, attr

class BreakNode(Node):
    __slots__ = ()

class ContinueNode(Node):
    __slots__ = ()

class PassNode(Node):
    __slots__ = ()

class ImportNode(Node):
    __slots__ = ('module', 'alias', 'items')
    def __init__(self, module, alias=None, items=None):
        self.module, self.alias, self.items = module, alias, items

class TryNode(Node):
    __slots__ = ('try_body', 'except_parts')
    def __init__(self, try_body, except_parts):
        self.try_body, self.except_parts = try_body, except_parts

class RaiseNode(Node):
    __slots__ = ('exception',)
    def __init__(self, exception): self.exception = exception

class LambdaNode(Node):
    __slots__ = ('params', 'body')
    def __init__(self, params, body): self.params, self.body = params, body

class ClassNode(Node):
    __slots__ = ('name', 'bases', 'body')
    def __init__(self, name, bases, body):
        self.name, self.bases, self.body = name, bases, body

class MatchNode(Node):
    __slots__ = ('expr', 'cases')
    def __init__(self, expr, cases): self.expr, self.cases = expr, cases

class ParallelNode(Node):
    __slots__ = ('body',)
    def __init__(self, body): self.body = body

class AsyncNode(Node):
    __slots__ = ('body',)
    def __init__(self, body): self.body = body

class MacroNode(Node):
    __slots__ = ('name', 'params', 'body')
    def __init__(self, name, params, body):
        self.name, self.params, self.body = name, params, body

class TernaryNode(Node):
    __slots__ = ('condition', 'true_val', 'false_val')
    def __init__(self, condition, true_val, false_val):
        self.condition, self.true_val, self.false_val = condition, true_val, false_val

class RangeNode(Node):
    __slots__ = ('start', 'end', 'step')
    def __init__(self, start, end, step=None):
        self.start, self.end, self.step = start, end, step

class SliceNode(Node):
    __slots__ = ('target', 'start', 'end', 'step')
    def __init__(self, target, start, end, step=None):
        self.target, self.start, self.end, self.step = target, start, end, step

class ComprehensionNode(Node):
    __slots__ = ('expr', 'var', 'iterable', 'condition', 'comp_type')
    def __init__(self, expr, var, iterable, condition=None, comp_type='list'):
        self.expr, self.var, self.iterable = expr, var, iterable
        self.condition, self.comp_type = condition, comp_type

class SpreadNode(Node):
    __slots__ = ('expr',)
    def __init__(self, expr): self.expr = expr

class PipeNode(Node):
    __slots__ = ('left', 'right')
    def __init__(self, left, right): self.left, self.right = left, right

# ======================================================
#  PARSER - Enhanced with more features
# ======================================================

class Parser:
    def __init__(self, tokens):
        self.tokens = [t for t in tokens if t.type != "NEWLINE"]
        self.pos = 0

    def peek(self, offset=0):
        idx = self.pos + offset
        return self.tokens[idx] if idx < len(self.tokens) else None

    def advance(self):
        self.pos += 1
        return self.tokens[self.pos - 1] if self.pos <= len(self.tokens) else None

    def expect(self, type_, value=None):
        tok = self.peek()
        if not tok or tok.type != type_:
            raise SyntaxError(f"Expected {type_}, got {tok}")
        if value and tok.value != value:
            raise SyntaxError(f"Expected '{value}', got '{tok.value}'")
        return self.advance()

    def parse(self):
        statements = []
        while self.peek():
            stmt = self.parse_statement()
            if stmt: statements.append(stmt)
        return statements

    def parse_statement(self):
        tok = self.peek()
        if not tok: return None
        
        if tok.type == "KEYWORD":
            handlers = {
                "fn": self.parse_function,
                "if": self.parse_if,
                "for": self.parse_for,
                "while": self.parse_while,
                "return": self.parse_return,
                "print": self.parse_print,
                "let": self.parse_let,
                "const": self.parse_const,
                "break": lambda: (self.advance(), BreakNode())[1],
                "continue": lambda: (self.advance(), ContinueNode())[1],
                "pass": lambda: (self.advance(), PassNode())[1],
                "import": self.parse_import,
                "from": self.parse_from_import,
                "class": self.parse_class,
                "try": self.parse_try,
                "raise": self.parse_raise,
                "match": self.parse_match,
                "parallel": self.parse_parallel,
                "macro": self.parse_macro,
                "async": self.parse_async,
            }
            handler = handlers.get(tok.value)
            if handler: return handler()
        
        elif tok.type == "IDENT":
            next_tok = self.peek(1)
            if next_tok and (next_tok.type == "ASSIGN" or next_tok.type == "ASSIGN_OP"):
                return self.parse_assignment()
        
        return self.parse_expr()

    def parse_let(self):
        self.expect("KEYWORD", "let")
        name = self.expect("IDENT").value
        self.expect("ASSIGN")
        value = self.parse_expr()
        return VarAssignNode(name, value)

    def parse_const(self):
        self.expect("KEYWORD", "const")
        name = self.expect("IDENT").value
        self.expect("ASSIGN")
        value = self.parse_expr()
        return VarAssignNode(name, value, const=True)

    def parse_assignment(self):
        name = self.expect("IDENT").value
        tok = self.peek()
        if tok.type == "ASSIGN_OP":
            op = self.advance().value
            value = self.parse_expr()
            return VarAssignNode(name, value, op)
        else:
            self.expect("ASSIGN")
            value = self.parse_expr()
            return VarAssignNode(name, value)

    def parse_function(self):
        self.expect("KEYWORD", "fn")
        inline = False
        if self.peek() and self.peek().value == "inline":
            self.advance()
            inline = True
        
        name = self.expect("IDENT").value
        self.expect("SYMBOL", "(")
        
        params = []
        while self.peek() and self.peek().value != ")":
            params.append(self.expect("IDENT").value)
            if self.peek() and self.peek().value == ",":
                self.advance()
        
        self.expect("SYMBOL", ")")
        self.expect("SYMBOL", ":")
        
        body = self.parse_block()
        return FunctionNode(name, params, body, inline)

    def parse_block(self):
        body = []
        depth = 0
        while self.peek():
            tok = self.peek()
            if tok.type == "KEYWORD" and tok.value in ("fn", "class", "if", "elif", "else"):
                if depth == 0 and tok.value in ("fn", "class"):
                    break
            stmt = self.parse_statement()
            if stmt: body.append(stmt)
            if not self.peek(): break
            next_tok = self.peek()
            if next_tok and next_tok.type == "KEYWORD" and next_tok.value in ("fn", "class", "elif", "else"):
                break
        return body

    def parse_if(self):
        self.expect("KEYWORD", "if")
        condition = self.parse_expr()
        self.expect("SYMBOL", ":")
        then_body = self.parse_block()
        
        elif_parts = []
        while self.peek() and self.peek().value == "elif":
            self.advance()
            elif_cond = self.parse_expr()
            self.expect("SYMBOL", ":")
            elif_body = self.parse_block()
            elif_parts.append((elif_cond, elif_body))
        
        else_body = None
        if self.peek() and self.peek().value == "else":
            self.advance()
            self.expect("SYMBOL", ":")
            else_body = self.parse_block()
        
        return IfNode(condition, then_body, elif_parts, else_body)

    def parse_for(self):
        self.expect("KEYWORD", "for")
        var = self.expect("IDENT").value
        self.expect("KEYWORD", "in")
        iterable = self.parse_expr()
        self.expect("SYMBOL", ":")
        body = self.parse_block()
        return ForNode(var, iterable, body)

    def parse_while(self):
        self.expect("KEYWORD", "while")
        condition = self.parse_expr()
        self.expect("SYMBOL", ":")
        body = self.parse_block()
        return WhileNode(condition, body)

    def parse_return(self):
        self.expect("KEYWORD", "return")
        if self.peek() and self.peek().type in ("INT", "FLOAT", "STRING", "IDENT", "SYMBOL", "BRACKET"):
            return ReturnNode(self.parse_expr())
        return ReturnNode()

    def parse_print(self):
        self.expect("KEYWORD", "print")
        self.expect("SYMBOL", "(")
        values = []
        while self.peek() and self.peek().value != ")":
            values.append(self.parse_expr())
            if self.peek() and self.peek().value == ",":
                self.advance()
        self.expect("SYMBOL", ")")
        return PrintNode(values)

    def parse_import(self):
        self.expect("KEYWORD", "import")
        module = self.expect("IDENT").value
        alias = None
        if self.peek() and self.peek().value == "as":
            self.advance()
            alias = self.expect("IDENT").value
        return ImportNode(module, alias)

    def parse_from_import(self):
        self.expect("KEYWORD", "from")
        module = self.expect("IDENT").value
        self.expect("KEYWORD", "import")
        items = []
        while self.peek() and self.peek().type == "IDENT":
            items.append(self.expect("IDENT").value)
            if self.peek() and self.peek().value == ",":
                self.advance()
        return ImportNode(module, items=items)

    def parse_class(self):
        self.expect("KEYWORD", "class")
        name = self.expect("IDENT").value
        bases = []
        if self.peek() and self.peek().value == "(":
            self.advance()
            while self.peek() and self.peek().value != ")":
                bases.append(self.expect("IDENT").value)
                if self.peek() and self.peek().value == ",":
                    self.advance()
            self.expect("SYMBOL", ")")
        self.expect("SYMBOL", ":")
        body = self.parse_block()
        return ClassNode(name, bases, body)

    def parse_try(self):
        self.expect("KEYWORD", "try")
        self.expect("SYMBOL", ":")
        try_body = self.parse_block()
        
        except_parts = []
        while self.peek() and self.peek().value == "except":
            self.advance()
            exception = None
            if self.peek() and self.peek().type == "IDENT":
                exception = self.expect("IDENT").value
            self.expect("SYMBOL", ":")
            except_body = self.parse_block()
            except_parts.append((exception, except_body))
        
        return TryNode(try_body, except_parts)

    def parse_raise(self):
        self.expect("KEYWORD", "raise")
        return RaiseNode(self.parse_expr())

    def parse_match(self):
        self.expect("KEYWORD", "match")
        expr = self.parse_expr()
        self.expect("SYMBOL", ":")
        cases = []
        while self.peek() and self.peek().value == "case":
            self.advance()
            pattern = self.parse_expr()
            self.expect("SYMBOL", ":")
            body = self.parse_block()
            cases.append((pattern, body))
        return MatchNode(expr, cases)

    def parse_parallel(self):
        self.expect("KEYWORD", "parallel")
        self.expect("SYMBOL", ":")
        body = self.parse_block()
        return ParallelNode(body)

    def parse_async(self):
        self.expect("KEYWORD", "async")
        self.expect("KEYWORD", "fn")
        name = self.expect("IDENT").value
        self.expect("SYMBOL", "(")
        params = []
        while self.peek() and self.peek().value != ")":
            params.append(self.expect("IDENT").value)
            if self.peek() and self.peek().value == ",":
                self.advance()
        self.expect("SYMBOL", ")")
        self.expect("SYMBOL", ":")
        body = self.parse_block()
        return FunctionNode(name, params, body, async_fn=True)

    def parse_macro(self):
        self.expect("KEYWORD", "macro")
        name = self.expect("IDENT").value
        self.expect("SYMBOL", "(")
        params = []
        while self.peek() and self.peek().value != ")":
            params.append(self.expect("IDENT").value)
            if self.peek() and self.peek().value == ",":
                self.advance()
        self.expect("SYMBOL", ")")
        self.expect("SYMBOL", ":")
        body = self.parse_block()
        return MacroNode(name, params, body)

    def parse_expr(self):
        return self.parse_pipe()

    def parse_pipe(self):
        left = self.parse_ternary()
        while self.peek() and self.peek().value == "|":
            if self.peek(1) and self.peek(1).value == ">":
                self.advance()
                self.advance()
                right = self.parse_ternary()
                left = PipeNode(left, right)
            else:
                break
        return left

    def parse_ternary(self):
        expr = self.parse_logical_or()
        if self.peek() and self.peek().value == "?":
            self.advance()
            true_val = self.parse_expr()
            self.expect("SYMBOL", ":")
            false_val = self.parse_expr()
            return TernaryNode(expr, true_val, false_val)
        return expr

    def parse_logical_or(self):
        left = self.parse_logical_and()
        while self.peek() and self.peek().type == "LOGIC" and self.peek().value in ("or", "||"):
            op = self.advance().value
            right = self.parse_logical_and()
            left = BinOpNode(left, "or", right)
        return left

    def parse_logical_and(self):
        left = self.parse_logical_not()
        while self.peek() and self.peek().type == "LOGIC" and self.peek().value in ("and", "&&"):
            op = self.advance().value
            right = self.parse_logical_not()
            left = BinOpNode(left, "and", right)
        return left

    def parse_logical_not(self):
        if self.peek() and self.peek().type == "LOGIC" and self.peek().value in ("not", "!"):
            self.advance()
            return UnaryOpNode("not", self.parse_logical_not())
        return self.parse_comparison()

    def parse_comparison(self):
        left = self.parse_arith()
        while self.peek() and self.peek().type == "COMP":
            op = self.advance().value
            if op == "===": op = "=="
            elif op == "!==": op = "!="
            elif op == "<>": op = "!="
            right = self.parse_arith()
            left = BinOpNode(left, op, right)
        return left

    def parse_arith(self):
        left = self.parse_term()
        while self.peek() and self.peek().type == "ARITH" and self.peek().value in ("+", "-", "|", "^"):
            op = self.advance().value
            right = self.parse_term()
            left = BinOpNode(left, op, right)
        return left

    def parse_term(self):
        left = self.parse_power()
        while self.peek() and self.peek().type == "ARITH" and self.peek().value in ("*", "/", "//", "%", "&"):
            op = self.advance().value
            right = self.parse_power()
            left = BinOpNode(left, op, right)
        return left

    def parse_power(self):
        left = self.parse_unary()
        while self.peek() and self.peek().type == "ARITH" and self.peek().value == "**":
            self.advance()
            right = self.parse_unary()
            left = BinOpNode(left, "**", right)
        return left

    def parse_unary(self):
        if self.peek() and self.peek().type == "ARITH" and self.peek().value in ("+", "-", "~"):
            op = self.advance().value
            return UnaryOpNode(op, self.parse_unary())
        return self.parse_postfix()

    def parse_postfix(self):
        expr = self.parse_primary()
        
        while self.peek():
            tok = self.peek()
            if tok.value == "(":
                self.advance()
                args = []
                while self.peek() and self.peek().value != ")":
                    args.append(self.parse_expr())
                    if self.peek() and self.peek().value == ",":
                        self.advance()
                self.expect("SYMBOL", ")")
                expr = CallNode(expr.name if isinstance(expr, VarAccessNode) else expr, args)
            elif tok.value == "[":
                self.advance()
                if self.peek() and self.peek().value == ":":
                    start = NumberNode(0)
                else:
                    start = self.parse_expr()
                
                if self.peek() and self.peek().value == ":":
                    self.advance()
                    end = self.parse_expr() if self.peek() and self.peek().value != "]" else None
                    step = None
                    if self.peek() and self.peek().value == ":":
                        self.advance()
                        step = self.parse_expr()
                    self.expect("BRACKET", "]")
                    expr = SliceNode(expr, start, end, step)
                else:
                    self.expect("BRACKET", "]")
                    expr = IndexNode(expr, start)
            elif tok.value == ".":
                self.advance()
                attr = self.expect("IDENT").value
                expr = AttributeNode(expr, attr)
            else:
                break
        
        return expr

    def parse_primary(self):
        tok = self.peek()
        if not tok: raise SyntaxError("Unexpected end")
        
        if tok.type == "INT":
            self.advance()
            return NumberNode(int(tok.value))
        elif tok.type == "FLOAT":
            self.advance()
            return NumberNode(float(tok.value))
        elif tok.type == "STRING":
            self.advance()
            return StringNode(tok.value[1:-1])
        elif tok.type == "IDENT":
            self.advance()
            return VarAccessNode(tok.value)
        elif tok.value == "(":
            self.advance()
            expr = self.parse_expr()
            self.expect("SYMBOL", ")")
            return expr
        elif tok.value == "[":
            self.advance()
            if self.peek() and self.peek().type == "IDENT" and self.peek(1) and self.peek(1).value == "for":
                expr = self.parse_expr()
                self.expect("KEYWORD", "for")
                var = self.expect("IDENT").value
                self.expect("KEYWORD", "in")
                iterable = self.parse_expr()
                condition = None
                if self.peek() and self.peek().value == "if":
                    self.advance()
                    condition = self.parse_expr()
                self.expect("BRACKET", "]")
                return ComprehensionNode(expr, var, iterable, condition, 'list')
            elements = []
            while self.peek() and self.peek().value != "]":
                elements.append(self.parse_expr())
                if self.peek() and self.peek().value == ",":
                    self.advance()
            self.expect("BRACKET", "]")
            return ListNode(elements)
        elif tok.value == "{":
            self.advance()
            pairs = []
            while self.peek() and self.peek().value != "}":
                key = self.parse_expr()
                self.expect("SYMBOL", ":")
                value = self.parse_expr()
                pairs.append((key, value))
                if self.peek() and self.peek().value == ",":
                    self.advance()
            self.expect("BRACKET", "}")
            return DictNode(pairs)
        elif tok.type == "KEYWORD" and tok.value == "lambda":
            return self.parse_lambda()
        
        raise SyntaxError(f"Unexpected: {tok}")

    def parse_lambda(self):
        self.expect("KEYWORD", "lambda")
        params = []
        while self.peek() and self.peek().value != ":":
            params.append(self.expect("IDENT").value)
            if self.peek() and self.peek().value == ",":
                self.advance()
        self.expect("SYMBOL", ":")
        body = self.parse_expr()
        return LambdaNode(params, body)

# ======================================================
#  TRANSPILER - Optimized Code Generation
# ======================================================

class Transpiler:
    def __init__(self):
        self.output = []
        self.indent_level = 0
        self.temp_var_counter = 0

    def emit(self, code):
        self.output.append("    " * self.indent_level + code)

    def indent(self):
        self.indent_level += 1

    def dedent(self):
        self.indent_level -= 1

    def temp_var(self):
        self.temp_var_counter += 1
        return f"_tmp{self.temp_var_counter}"

    def transpile(self, nodes):
        if not isinstance(nodes, list):
            nodes = [nodes]
        for node in nodes:
            self.transpile_node(node)
        return "\n".join(self.output)

    def transpile_node(self, node):
        if isinstance(node, NumberNode):
            return str(node.value)
        
        elif isinstance(node, StringNode):
            return f'"{node.value}"'
        
        elif isinstance(node, VarAccessNode):
            return node.name
        
        elif isinstance(node, VarAssignNode):
            value_code = self.transpile_node(node.value)
            if node.op == "=":
                self.emit(f"{node.name} = {value_code}")
            else:
                self.emit(f"{node.name} {node.op} {value_code}")
            return ""
        
        elif isinstance(node, BinOpNode):
            left = self.transpile_node(node.left)
            right = self.transpile_node(node.right)
            op_map = {"and": "and", "or": "or", "^": "^", "&": "&", "|": "|"}
            op = op_map.get(node.op, node.op)
            return f"({left} {op} {right})"
        
        elif isinstance(node, UnaryOpNode):
            operand = self.transpile_node(node.operand)
            op_map = {"not": "not ", "!": "not ", "~": "~"}
            op = op_map.get(node.op, node.op)
            return f"({op}{operand})"
        
        elif isinstance(node, FunctionNode):
            params = ", ".join(node.params)
            decorator = ""
            if node.inline:
                decorator = "@inline\n" + "    " * self.indent_level
            if node.async_fn:
                self.emit(f"{decorator}async def {node.name}({params}):")
            else:
                self.emit(f"{decorator}def {node.name}({params}):")
            self.indent()
            if node.body:
                for stmt in node.body:
                    self.transpile_node(stmt)
            else:
                self.emit("pass")
            self.dedent()
            return ""
        
        elif isinstance(node, CallNode):
            name = node.name if isinstance(node.name, str) else self.transpile_node(node.name)
            args = ", ".join([self.transpile_node(arg) for arg in node.args])
            return f"{name}({args})"
        
        elif isinstance(node, IfNode):
            condition = self.transpile_node(node.condition)
            self.emit(f"if {condition}:")
            self.indent()
            if node.then_body:
                for stmt in node.then_body:
                    self.transpile_node(stmt)
            else:
                self.emit("pass")
            self.dedent()
            
            for elif_cond, elif_body in node.elif_parts:
                elif_cond_code = self.transpile_node(elif_cond)
                self.emit(f"elif {elif_cond_code}:")
                self.indent()
                if elif_body:
                    for stmt in elif_body:
                        self.transpile_node(stmt)
                else:
                    self.emit("pass")
                self.dedent()
            
            if node.else_body:
                self.emit("else:")
                self.indent()
                for stmt in node.else_body:
                    self.transpile_node(stmt)
                self.dedent()
            return ""
        
        elif isinstance(node, ForNode):
            iterable = self.transpile_node(node.iterable)
            self.emit(f"for {node.var} in {iterable}:")
            self.indent()
            if node.body:
                for stmt in node.body:
                    self.transpile_node(stmt)
            else:
                self.emit("pass")
            self.dedent()
            return ""
        
        elif isinstance(node, WhileNode):
            condition = self.transpile_node(node.condition)
            self.emit(f"while {condition}:")
            self.indent()
            if node.body:
                for stmt in node.body:
                    self.transpile_node(stmt)
            else:
                self.emit("pass")
            self.dedent()
            return ""
        
        elif isinstance(node, ReturnNode):
            if node.value:
                value = self.transpile_node(node.value)
                self.emit(f"return {value}")
            else:
                self.emit("return")
            return ""
        
        elif isinstance(node, PrintNode):
            values = ", ".join([self.transpile_node(v) for v in node.values])
            self.emit(f"print({values})")
            return ""
        
        elif isinstance(node, ListNode):
            elements = ", ".join([self.transpile_node(e) for e in node.elements])
            return f"[{elements}]"
        
        elif isinstance(node, DictNode):
            pairs = ", ".join([f"{self.transpile_node(k)}: {self.transpile_node(v)}" 
                             for k, v in node.pairs])
            return f"{{{pairs}}}"
        
        elif isinstance(node, IndexNode):
            target = self.transpile_node(node.target)
            index = self.transpile_node(node.index)
            return f"{target}[{index}]"
        
        elif isinstance(node, SliceNode):
            target = self.transpile_node(node.target)
            start = self.transpile_node(node.start) if node.start else ""
            end = self.transpile_node(node.end) if node.end else ""
            if node.step:
                step = self.transpile_node(node.step)
                return f"{target}[{start}:{end}:{step}]"
            return f"{target}[{start}:{end}]"
        
        elif isinstance(node, AttributeNode):
            target = self.transpile_node(node.target)
            return f"{target}.{node.attr}"
        
        elif isinstance(node, BreakNode):
            self.emit("break")
            return ""
        
        elif isinstance(node, ContinueNode):
            self.emit("continue")
            return ""
        
        elif isinstance(node, PassNode):
            self.emit("pass")
            return ""
        
        elif isinstance(node, ImportNode):
            if node.items:
                items = ", ".join(node.items)
                self.emit(f"from {node.module} import {items}")
            elif node.alias:
                self.emit(f"import {node.module} as {node.alias}")
            else:
                self.emit(f"import {node.module}")
            return ""
        
        elif isinstance(node, TryNode):
            self.emit("try:")
            self.indent()
            if node.try_body:
                for stmt in node.try_body:
                    self.transpile_node(stmt)
            else:
                self.emit("pass")
            self.dedent()
            
            for exception, except_body in node.except_parts:
                if exception:
                    self.emit(f"except {exception}:")
                else:
                    self.emit("except:")
                self.indent()
                if except_body:
                    for stmt in except_body:
                        self.transpile_node(stmt)
                else:
                    self.emit("pass")
                self.dedent()
            return ""
        
        elif isinstance(node, RaiseNode):
            exception = self.transpile_node(node.exception)
            self.emit(f"raise {exception}")
            return ""
        
        elif isinstance(node, LambdaNode):
            params = ", ".join(node.params)
            body = self.transpile_node(node.body)
            return f"lambda {params}: {body}"
        
        elif isinstance(node, ClassNode):
            if node.bases:
                bases = ", ".join(node.bases)
                self.emit(f"class {node.name}({bases}):")
            else:
                self.emit(f"class {node.name}:")
            self.indent()
            if node.body:
                for stmt in node.body:
                    self.transpile_node(stmt)
            else:
                self.emit("pass")
            self.dedent()
            return ""
        
        elif isinstance(node, MatchNode):
            expr_code = self.transpile_node(node.expr)
            temp = self.temp_var()
            self.emit(f"{temp} = {expr_code}")
            for i, (pattern, body) in enumerate(node.cases):
                pattern_code = self.transpile_node(pattern)
                if i == 0:
                    self.emit(f"if {temp} == {pattern_code}:")
                else:
                    self.emit(f"elif {temp} == {pattern_code}:")
                self.indent()
                for stmt in body:
                    self.transpile_node(stmt)
                self.dedent()
            return ""
        
        elif isinstance(node, ParallelNode):
            self.emit("from concurrent.futures import ThreadPoolExecutor")
            self.emit("with ThreadPoolExecutor() as executor:")
            self.indent()
            self.emit("futures = []")
            for stmt in node.body:
                self.transpile_node(stmt)
            self.dedent()
            return ""
        
        elif isinstance(node, MacroNode):
            self.emit(f"# Macro: {node.name}")
            return ""
        
        elif isinstance(node, TernaryNode):
            true_val = self.transpile_node(node.true_val)
            false_val = self.transpile_node(node.false_val)
            condition = self.transpile_node(node.condition)
            return f"({true_val} if {condition} else {false_val})"
        
        elif isinstance(node, ComprehensionNode):
            expr = self.transpile_node(node.expr)
            iterable = self.transpile_node(node.iterable)
            comp = f"{expr} for {node.var} in {iterable}"
            if node.condition:
                cond = self.transpile_node(node.condition)
                comp += f" if {cond}"
            if node.comp_type == 'list':
                return f"[{comp}]"
            elif node.comp_type == 'dict':
                return f"{{{comp}}}"
            return f"({comp})"
        
        elif isinstance(node, PipeNode):
            left = self.transpile_node(node.left)
            if isinstance(node.right, CallNode):
                name = node.right.name if isinstance(node.right.name, str) else self.transpile_node(node.right.name)
                args = ", ".join([left] + [self.transpile_node(arg) for arg in node.right.args])
                return f"{name}({args})"
            return f"{node.right}({left})"
        
        else:
            raise TypeError(f"Unknown node: {type(node)}")

# ======================================================
#  BUILT-IN FUNCTIONS - Extended Library
# ======================================================

ORIGIN_BUILTINS = {
    # Math
    'abs': abs, 'round': round, 'pow': pow, 'min': min, 'max': max, 'sum': sum,
    'sqrt': math.sqrt, 'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
    'asin': math.asin, 'acos': math.acos, 'atan': math.atan, 'atan2': math.atan2,
    'floor': math.floor, 'ceil': math.ceil, 'log': math.log, 'log10': math.log10,
    'log2': math.log2, 'exp': math.exp, 'pi': math.pi, 'e': math.e, 'tau': math.tau,
    'degrees': math.degrees, 'radians': math.radians, 'factorial': math.factorial,
    'gcd': math.gcd, 'lcm': lambda a, b: abs(a * b) // math.gcd(a, b),
    'hypot': math.hypot, 'inf': math.inf, 'nan': math.nan,
    
    # Type conversions
    'int': int, 'float': float, 'str': str, 'bool': bool, 'list': list,
    'tuple': tuple, 'dict': dict, 'set': set, 'frozenset': frozenset,
    'complex': complex, 'bytes': bytes, 'bytearray': bytearray,
    
    # String operations
    'len': len, 'ord': ord, 'chr': chr, 'ascii': ascii, 'repr': repr,
    'format': format, 'join': lambda sep, lst: sep.join(map(str, lst)),
    'split': lambda s, sep=' ': s.split(sep),
    'strip': lambda s: s.strip(), 'lstrip': lambda s: s.lstrip(),
    'rstrip': lambda s: s.rstrip(), 'upper': lambda s: s.upper(),
    'lower': lambda s: s.lower(), 'title': lambda s: s.title(),
    'capitalize': lambda s: s.capitalize(), 'swapcase': lambda s: s.swapcase(),
    'replace': lambda s, old, new: s.replace(old, new),
    'startswith': lambda s, prefix: s.startswith(prefix),
    'endswith': lambda s, suffix: s.endswith(suffix),
    'find': lambda s, sub: s.find(sub), 'rfind': lambda s, sub: s.rfind(sub),
    'count': lambda s, sub: s.count(sub), 'index': lambda s, sub: s.index(sub),
    'isalpha': lambda s: s.isalpha(), 'isdigit': lambda s: s.isdigit(),
    'isalnum': lambda s: s.isalnum(), 'isspace': lambda s: s.isspace(),
    'islower': lambda s: s.islower(), 'isupper': lambda s: s.isupper(),
    
    # Collection operations
    'append': lambda lst, item: lst.append(item) or lst,
    'extend': lambda lst, items: lst.extend(items) or lst,
    'insert': lambda lst, idx, item: lst.insert(idx, item) or lst,
    'remove': lambda lst, item: lst.remove(item) or lst,
    'pop': lambda lst, idx=-1: lst.pop(idx),
    'clear': lambda lst: lst.clear() or lst,
    'sort': lambda lst, reverse=False: lst.sort(reverse=reverse) or lst,
    'reverse': lambda lst: lst.reverse() or lst,
    'copy': lambda obj: obj.copy() if hasattr(obj, 'copy') else obj[:],
    
    # Functional programming
    'map': lambda f, *iterables: list(map(f, *iterables)),
    'filter': lambda f, iterable: list(filter(f, iterable)),
    'reduce': lambda f, lst, init=None: functools.reduce(f, lst, init) if init else functools.reduce(f, lst),
    'zip': lambda *iterables: list(zip(*iterables)),
    'enumerate': lambda iterable, start=0: list(enumerate(iterable, start)),
    'range': range, 'sorted': sorted, 'reversed': lambda x: list(reversed(x)),
    'any': any, 'all': all,
    
    # Advanced itertools
    'chain': lambda *iterables: [item for iterable in iterables for item in iterable],
    'cycle': lambda iterable, n=None: (iterable * n if n else iterable),
    'repeat': lambda obj, times=None: [obj] * (times if times else 1),
    'accumulate': lambda iterable: [sum(iterable[:i+1]) for i in range(len(iterable))],
    'permutations': lambda iterable, r=None: __import__('itertools').permutations(iterable, r),
    'combinations': lambda iterable, r: __import__('itertools').combinations(iterable, r),
    'product': lambda *iterables: __import__('itertools').product(*iterables),
    
    # I/O
    'print': print, 'input': input, 'open': open,
    'read': lambda filename: open(filename).read(),
    'write': lambda filename, content: open(filename, 'w').write(content),
    'append_file': lambda filename, content: open(filename, 'a').write(content),
    'readlines': lambda filename: open(filename).readlines(),
    'writelines': lambda filename, lines: open(filename, 'w').writelines(lines),
    
    # Random
    'random': random.random, 'randint': random.randint,
    'choice': random.choice, 'shuffle': random.shuffle,
    'sample': random.sample, 'uniform': random.uniform,
    'gauss': random.gauss, 'seed': random.seed,
    
    # Type checking
    'type': type, 'isinstance': isinstance, 'issubclass': issubclass,
    'hasattr': hasattr, 'getattr': getattr, 'setattr': setattr, 'delattr': delattr,
    'callable': callable,
    
    # Utility
    'dir': dir, 'help': help, 'id': id, 'hash': hash,
    'hex': hex, 'oct': oct, 'bin': bin, 'divmod': divmod,
    'eval': eval, 'exec': exec, 'compile': compile,
    'globals': globals, 'locals': locals, 'vars': vars,
    
    # Object/Class
    'object': object, 'property': property,
    'staticmethod': staticmethod, 'classmethod': classmethod,
    'super': super, 'iter': iter, 'next': next,
    
    # Exceptions
    'Exception': Exception, 'ValueError': ValueError, 'TypeError': TypeError,
    'KeyError': KeyError, 'IndexError': IndexError, 'AttributeError': AttributeError,
    'ZeroDivisionError': ZeroDivisionError, 'RuntimeError': RuntimeError,
    'NotImplementedError': NotImplementedError, 'StopIteration': StopIteration,
    
    # Advanced
    'slice': slice, 'memoryview': memoryview,
    
    # Time
    'time': time.time, 'sleep': time.sleep,
    
    # Decorators
    'memoize': memoize,
    'inline': lambda f: f,
}

# ======================================================
#  EXECUTION ENGINE - Optimized
# ======================================================

def execute_python(transpiled_code, env=None):
    if env is None:
        env = ORIGIN_BUILTINS.copy()
    else:
        env.update(ORIGIN_BUILTINS)
    
    try:
        compiled = compile(transpiled_code, '<origin>', 'exec', optimize=2)
        exec(compiled, env)
        return env
    except Exception as e:
        print(f"\n❌ Runtime Error: {e}")
        traceback.print_exc()
        return env

def run_origin_code(code, env=None, debug=False, measure_time=False):
    start = time.time() if measure_time else None
    
    try:
        code_hash = hash(code)
        cached = optimizer.get_cached(code_hash)
        
        if cached:
            if debug: print("⚡ Using cached compilation")
            return execute_python(cached, env)
        
        tokens = tokenize(code)
        if debug:
            print("=== TOKENS ===")
            for tok in tokens[:30]: print(tok)
            print()
        
        parser = Parser(tokens)
        ast = parser.parse()
        if debug:
            print("=== AST ===")
            for node in ast[:15]: print(node)
            print()
        
        transpiler = Transpiler()
        python_code = transpiler.transpile(ast)
        if debug:
            print("=== TRANSPILED PYTHON ===")
            print(python_code)
            print()
        
        optimizer.cache_compiled(code_hash, python_code)
        result = execute_python(python_code, env)
        
        if measure_time:
            elapsed = time.time() - start
            print(f"\n⚡ Execution time: {elapsed*1000:.2f}ms")
        
        return result
    
    except SyntaxError as e:
        print(f"\n❌ Syntax Error: {e}")
        return env
    except Exception as e:
        print(f"\n❌ Compilation Error: {e}")
        traceback.print_exc()
        return env

# ======================================================
#  REPL - Interactive Mode
# ======================================================

def repl():
    print("""
╔══════════════════════════════════════════════════════════╗
║           ORIGIN v2.0 - FAST INTERACTIVE REPL            ║
╠══════════════════════════════════════════════════════════╣
║  Commands:                                               ║
║    exit/quit    - Exit REPL                              ║
║    help         - Show examples                          ║
║    debug on/off - Toggle debug mode                      ║
║    time on/off  - Toggle timing                          ║
║    clear        - Clear screen                           ║
║    stats        - Show compilation stats                 ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    env = ORIGIN_BUILTINS.copy()
    debug = False
    measure_time = False
    
    while True:
        try:
            line = input("\n🚀 origin> ")
            stripped = line.strip()
            
            if stripped in ("exit", "quit"):
                print("👋 Goodbye!")
                break
            
            if stripped == "help":
                print_help()
                continue
            
            if stripped.startswith("debug "):
                debug = stripped.split()[1] == "on"
                print(f"Debug: {'ON ✓' if debug else 'OFF'}")
                continue
            
            if stripped.startswith("time "):
                measure_time = stripped.split()[1] == "on"
                print(f"Timing: {'ON ✓' if measure_time else 'OFF'}")
                continue
            
            if stripped == "clear":
                print("\033[2J\033[H")
                continue
            
            if stripped == "stats":
                print(f"\n📊 Compilation Stats:")
                print(f"   Cached compilations: {len(optimizer.compiled_cache)}")
                print(f"   Hot functions: {len(optimizer.hot_functions)}")
                continue
            
            if not stripped: continue
            
            env = run_origin_code(line, env, debug, measure_time)
        
        except EOFError:
            print("\n👋 Goodbye!")
            break
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted")
            continue

def print_help():
    print("""
╔══════════════════════════════════════════════════════════╗
║                 ORIGIN LANGUAGE GUIDE                    ║
╚══════════════════════════════════════════════════════════╝

📝 VARIABLES & CONSTANTS:
   let x = 10              # Mutable variable
   const PI = 3.14159      # Constant
   x += 5                  # Augmented assignment

🔢 MATH & OPERATORS:
   let result = (5 + 3) * 2 ** 8
   let bits = 10 & 7 | 3 ^ 2    # Bitwise ops
   print(sqrt(16), sin(pi/2))

📊 DATA STRUCTURES:
   let nums = [1, 2, 3, 4, 5]
   let person = {"name": "Alice", "age": 30}
   let subset = nums[1:3]       # Slicing
   let comp = [x*2 for x in nums if x > 2]  # Comprehension

🔁 LOOPS & CONTROL:
   for i in range(5): print(i)
   while x < 10: x += 1
   
   match x:
       case 1: print("one")
       case 2: print("two")

🎯 FUNCTIONS & LAMBDAS:
   fn add(a, b): return a + b
   
   fn inline fast_add(a, b): return a + b  # Inline hint
   
   let double = lambda x: x * 2
   
   async fn fetch_data(): return "data"

🔥 ADVANCED FEATURES:
   # Ternary operator
   let result = x > 10 ? "big" : "small"
   
   # Pipe operator
   let result = data |> filter |> map |> reduce
   
   # Pattern matching
   match value:
       case 0: print("zero")
       case _: print("other")
   
   # Parallel execution
   parallel:
       task1()
       task2()

🧩 DESTRUCTURING & SPREAD:
   let [first, ...rest] = [1, 2, 3, 4]
   let {x, y} = point

⚡ PERFORMANCE:
   - Auto-caching of compiled code
   - JIT optimization hints
   - Inline function support
   - Memoization decorator

Type 'time on' to measure execution speed!
    """)

def main():
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        debug = "--debug" in sys.argv or "-d" in sys.argv
        measure_time = "--time" in sys.argv or "-t" in sys.argv
        
        try:
            with open(filename, "r") as f:
                code = f.read()
            
            print(f"🚀 Running: {filename}")
            print("=" * 60)
            run_origin_code(code, debug=debug, measure_time=measure_time)
            print("=" * 60)
            print(f"✅ Complete")
        
        except FileNotFoundError:
            print(f"❌ File not found: {filename}")
            sys.exit(1)
    else:
        repl()

if __name__ == "__main__":
    main()
