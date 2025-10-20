# ======= LEXER =======
import re

KEYWORDS = {"let", "fn", "if", "elif", "else", "for", "while", "return", "in", "print", "abs", "sqrt", "pwr"}

class Token:
    def __init__(self, type_, value=None, line=0, col=0):
        self.type = type_
        self.value = value
        self.line = line
        self.col = col
    def __repr__(self):
        return f"Token({self.type}, {repr(self.value)})"

class Lexer:
    def __init__(self, lines):
        self.lines = lines
        self.tokens = []
        self.indent_stack = [0]

    def lex(self):
        for line_num, raw_line in enumerate(self.lines, start=1):
            line = raw_line.rstrip("\n")
            if not line.strip():
                continue

            # ---- Indentation tracking ----
            indent = len(line) - len(line.lstrip(" "))
            if indent > self.indent_stack[-1]:
                self.indent_stack.append(indent)
                self.tokens.append(Token("INDENT", None, line_num))
            while indent < self.indent_stack[-1]:
                self.indent_stack.pop()
                self.tokens.append(Token("DEDENT", None, line_num))

            # ---- Tokenize the actual code ----
            self.tokenize_line(line.strip(), line_num)

        # close all open indents
        while len(self.indent_stack) > 1:
            self.indent_stack.pop()
            self.tokens.append(Token("DEDENT"))

        return self.tokens

    def tokenize_line(self, line, line_num):
        token_spec = [
            ("NUMBER",   r"\d+(\.\d+)?"),
            ("STRING",   r"\".*?\"|'.*?'"),
            ("OP",       r"==|!=|<=|>=|[+\-*/%=<>]"),
            ("LPAREN",   r"\("),
            ("RPAREN",   r"\)"),
            ("COLON",    r":"),
            ("COMMA",    r","),
            ("IDENT",    r"[A-Za-z_][A-Za-z0-9_]*"),
        ]
        tok_regex = "|".join(f"(?P<{name}>{pattern})" for name, pattern in token_spec)
        for match in re.finditer(tok_regex, line):
            kind = match.lastgroup
            value = match.group()
            col = match.start() + 1

            if kind == "IDENT" and value in KEYWORDS:
                kind = "KEYWORD"
            self.tokens.append(Token(kind, value, line_num, col))
        self.tokens.append(Token("NEWLINE", None, line_num))

# ======= PARSER =======
class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def advance(self):
        self.pos += 1
        return self.peek()

    def skip_newlines(self):
        tok = self.peek()
        while tok and tok.type == "NEWLINE":
            self.advance()
            tok = self.peek()

    def parse(self):
        ast = []
        while self.peek() is not None:
            node = self.parse_statement()
            if node:
                ast.append(node)
        return ast

    def parse_statement(self):
        self.skip_newlines()
        tok = self.peek()

        if tok is None:
            return None

        if tok.type == "KEYWORD" and tok.value == "let":
            return self.parse_var_declaration()
        elif tok.type == "KEYWORD" and tok.value == "if":
            return self.parse_if_statement()
        elif tok.type == "KEYWORD" and tok.value == "print":
            return self.parse_print_statement()
        elif tok.type == "IDENT":
            return self.parse_expression()
        elif tok.type in ("DEDENT", "INDENT"):
            return None
        else:
            self.error("Unexpected token", tok)

    def error(self, message, token):
        raise SyntaxError(f"{message} at line {token.line}, column {token.col}")

    # ====== Statements ======
    def parse_var_declaration(self):
        self.advance()  # skip 'let'

        tok = self.peek()
        if tok is None:
            raise SyntaxError("Unexpected end of input after 'let'")

        if tok.type != "IDENT":
            self.error("Expected variable name", tok)

        name = tok.value
        self.advance()  # move past variable name

        self.expect("OP", "=")  # make sure next token is '='
        expr = self.parse_expression()
        self.skip_newlines()
        return ("var_decl", name, expr)

    def parse_if_statement(self):
        self.advance()  # skip 'if'
        condition = self.parse_expression()
        self.expect("COLON", ":")
        self.skip_newlines()
        body = self.parse_block()
        return ("if", condition, body)

    def parse_print_statement(self):
        self.advance()  # skip 'print'
        self.expect("LPAREN")
        expr = self.parse_expression()
        self.expect("RPAREN")
        self.skip_newlines()
        return ("print", expr)

    # ====== Expressions ======
    def parse_expression(self):
        left = self.parse_term()
        tok = self.peek()
        while tok and tok.type == "OP" and tok.value in ("+", "-", "==", "!=", ">", "<", ">=", "<="):
            op = tok.value
            self.advance()
            right = self.parse_term()
            left = ("binop", op, left, right)
            tok = self.peek()
        return left

    def parse_term(self):
        tok = self.peek()
        if tok and tok.type == "NUMBER":
            self.advance()
            return ("number", float(tok.value))
        elif tok and tok.type == "STRING":
            self.advance()
            return ("string", tok.value[1:-1])
        elif tok and tok.type == "IDENT":
            self.advance()
            return ("var", tok.value)
        elif tok and tok.type == "LPAREN":
            self.advance()
            expr = self.parse_expression()
            self.expect("RPAREN")
            return expr
        else:
            self.error("Unexpected token in expression", tok)

    # ====== Blocks ======
    def parse_block(self):
        self.expect("INDENT")
        self.skip_newlines()
        body = []
        tok = self.peek()
        while tok and tok.type != "DEDENT":
            stmt = self.parse_statement()
            if stmt is not None:
                body.append(stmt)
        self.expect("DEDENT")
        return body

    # ====== Utilities ======
    def expect(self, type_, value=None):
        tok = self.peek()
        if tok is None or tok.type != type_ or (value is not None and tok.value != value):
            self.error(f"Expected {type_} {value}", tok if tok else Token("EOF", line=0, col=0))
        self.advance()
        return tok

# ======= INTERPRETER =======
class Interpreter:
    def __init__(self):
        self.env = {}

    def evaluate(self, node):
        node_type = node[0]

        if node_type == "number":
            return node[1]
        if node_type == "string":
            return node[1]

        if node_type == "var":
            name = node[1]
            if name in self.env:
                return self.env[name]
            else:
                raise NameError(f"Variable '{name}' not defined")

        if node_type == "var_decl":
            name = node[1]
            value = self.evaluate(node[2])
            self.env[name] = value
            return value

        if node_type == "binop":
            left = self.evaluate(node[2])
            right = self.evaluate(node[3])
            return self.eval_binop(left, node[1], right)

        if node_type == "print":
            value = self.evaluate(node[1])
            print(value)
            return value

        if node_type == "if":
            condition = node[1]
            body = node[2]
            if self.evaluate(condition):
                return self.eval_block(body)
            return None

        raise Exception(f"Unknown node type: {node_type}")

    def eval_binop(self, left, op, right):
        if op == "+": return left + right
        if op == "-": return left - right
        if op == "*": return left * right
        if op == "/": return left / right
        if op == "%": return left % right
        if op == "==": return left == right
        if op == "!=": return left != right
        if op == ">": return left > right
        if op == "<": return left < right
        if op == ">=": return left >= right
        if op == "<=": return left <= right
        raise Exception(f"Unknown operator: {op}")

    def eval_block(self, block):
        result = None
        for stmt in block:
            result = self.evaluate(stmt)
        return result

# ======= RUNNER =======
if __name__ == "__main__":
    with open("code.txt", "r") as file:
        code = [line.rstrip("\n") for line in file]

    lexer = Lexer(code)
    tokens = lexer.lex()

    print("=== TOKENS ===")
    for token in tokens:
        print(token)
    
    print("\n=== EXECUTION ===")
    parser = Parser(tokens)
    ast = parser.parse()

    interpreter = Interpreter()
    for node in ast:
        interpreter.evaluate(node)
