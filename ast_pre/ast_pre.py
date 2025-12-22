import ast
from typing import Dict, Any

def attach_parent_pointers(tree: ast.AST):
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            child.parent = parent

def extract_ast_structure(CODE: Dict[str, Any]) -> Dict[str, Any]:
    source_code = CODE["solution_function"]
    # print(source_code)
    tree = ast.parse(source_code)
    attach_parent_pointers(tree)

    processor = ASTPreprocessor()
    processor.visit(tree)
    return processor.result

class ASTPreprocessor(ast.NodeVisitor):
    def __init__(self):
        self.result: Dict[str, Any] = {
            "imports": [],
            "ast_structure": []
        }
        self._current_function = None

    # ---------- import 处理 ----------
    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            if alias.name not in self.result["imports"]:
                self.result["imports"].append(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        module = node.module
        for alias in node.names:
            full_name = f"{module}.{alias.name}"
            if full_name not in self.result["imports"]:
                self.result["imports"].append(full_name)
        self.generic_visit(node)

    # ---------- 函数 ----------
    def visit_FunctionDef(self, node: ast.FunctionDef):
        func_info = {
            "function_name": node.name,
            "lineno": node.lineno,
            "api_calls": []
        }
        self.result["ast_structure"].append(func_info)
        self._current_function = func_info
        self.generic_visit(node)
        self._current_function = None

    # ---------- API 调用 ----------
    def visit_Call(self, node: ast.Call):
        api_name = self._resolve_call_name(node.func)
        if api_name and self._current_function is not None:
            self._current_function["api_calls"].append({
                "api": api_name,
                "lineno": node.lineno,
                "context": self._infer_context(node)
            })
        self.generic_visit(node)

    # ---------- 工具函数 ----------
    def _resolve_call_name(self, node):
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            parts.append(node.id)
        return ".".join(reversed(parts)) if parts else None

    def _infer_context(self, node):
        parent = getattr(node, "parent", None)
        if isinstance(parent, ast.While):
            return "while-condition"
        if isinstance(parent, ast.If):
            return "if-condition"
        return "expression"


