import javalang
from typing import Dict, Any, List


def extract_java_ast_structure(CODE: Dict[str, Any]) -> Dict[str, Any]:
    source_code = CODE["java_code"]

    tree = javalang.parse.parse(source_code)

    processor = JavaASTPreprocessor()
    processor.visit(tree)

    return processor.result


class JavaASTPreprocessor:
    def __init__(self):
        self.result: Dict[str, Any] = {
            "imports": [],
            "ast_structure": []
        }
        self._current_method = None
        self._visited = set()  # 跟踪已访问的节点，避免死循环

    # ---------- visitor 入口 ----------
    def visit(self, node):
        # 如果已经访问过该节点，则跳过
        node_id = id(node)
        if node_id in self._visited:
            return
        self._visited.add(node_id)
        
        method_name = f"visit_{type(node).__name__}"
        visitor = getattr(self, method_name, self.generic_visit)
        visitor(node)

    def generic_visit(self, node):
        for _, child in node:
            if isinstance(child, javalang.ast.Node):
                self.visit(child)
            elif isinstance(child, list):
                for item in child:
                    if isinstance(item, javalang.ast.Node):
                        self.visit(item)

    # ---------- import ----------
    def visit_Import(self, node: javalang.tree.Import):
        if node.path not in self.result["imports"]:
            self.result["imports"].append(node.path)

    # ---------- class ----------
    def visit_ClassDeclaration(self, node):
        self.generic_visit(node)

    # ---------- method ----------
    def visit_MethodDeclaration(self, node: javalang.tree.MethodDeclaration):
        func_info = {
            "function_name": node.name,
            "lineno": node.position.line if node.position else None,
            "api_calls": []
        }
        self.result["ast_structure"].append(func_info)
        self._current_method = func_info

        self.generic_visit(node)

        self._current_method = None

    # ---------- API 调用 ----------
    def visit_MethodInvocation(self, node: javalang.tree.MethodInvocation):
        if self._current_method is None:
            return

        api_name = self._resolve_call_name(node)
        self._current_method["api_calls"].append({
            "api": api_name,
            "lineno": node.position.line if node.position else None,
            "context": self._infer_context(node)
        })

        self.generic_visit(node)

    # ---------- System.out.println 特判 ----------
    def visit_MemberReference(self, node):
        self.generic_visit(node)

    # ---------- 工具函数 ----------
    def _resolve_call_name(self, node: javalang.tree.MethodInvocation) -> str:
        if node.qualifier:
            return f"{node.qualifier}.{node.member}"
        return node.member

    def _infer_context(self, node):
        parent = getattr(node, "parent", None)
        if isinstance(parent, javalang.tree.IfStatement):
            return "if-condition"
        if isinstance(parent, javalang.tree.WhileStatement):
            return "while-condition"
        return "expression"
