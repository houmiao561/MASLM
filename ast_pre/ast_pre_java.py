import javalang
from typing import Dict, Any


def extract_java_ast_structure(CODE: Dict[str, Any]) -> Dict[str, Any]:
    source_code = CODE["java_code"]

    tree = javalang.parse.parse(source_code)

    processor = JavaASTPreprocessor()
    processor.visit(tree)

    return processor.result


class JavaASTPreprocessor:
    def __init__(self):
        self.result: Dict[str, Any] = {
            "imports": {},          # simpleName -> fullName
            "ast_structure": []
        }
        self._current_method = None
        self._current_locals = {}  # varName -> fullType
        self._visited = set()

    # ---------- visitor 入口 ----------
    def visit(self, node):
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
        # e.g. com.sun.jarsigner.ContentSignerParameters
        full = node.path
        simple = full.split(".")[-1]
        self.result["imports"][simple] = full

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

        # 构建参数类型表
        self._current_locals = {}
        for param in node.parameters:
            if param.type:
                type_name = self._resolve_type(param.type)
                self._current_locals[param.name] = type_name

        self.generic_visit(node)

        self._current_method = None
        self._current_locals = {}

    # ---------- API 调用 ----------
    def visit_MethodInvocation(self, node: javalang.tree.MethodInvocation):
        if self._current_method is None:
            return

        api_fqn = self._resolve_api_fqn(node)

        self._current_method["api_calls"].append({
            "api": api_fqn,
            "lineno": node.position.line if node.position else None,
            "context": self._infer_context(node)
        })

        self.generic_visit(node)

    # ---------- 工具函数 ----------
    def _resolve_type(self, type_node):
        """
        将类型节点解析为全限定类名（如果能）
        """
        name = type_node.name
        return self.result["imports"].get(name, name)
    
    def _resolve_api_fqn(self, node: javalang.tree.MethodInvocation) -> str:
        """
        parameters.getSignatureAlgorithm
        -> com.sun.jarsigner.ContentSignerParameters.getSignatureAlgorithm

        System.out.println
        -> java.lang.System.out.println
        """
        if node.qualifier:
            qualifier = node.qualifier

            # 情况 1：参数 / 局部变量
            if qualifier in self._current_locals:
                return f"{self._current_locals[qualifier]}.{node.member}"

            # 情况 2：System.out.println / 类似链式访问
            if "." in qualifier:
                head = qualifier.split(".")[0]

                # System.out.println
                if head == "System":
                    return f"java.lang.{qualifier}.{node.member}"

            # 情况 3：直接类名静态调用
            if qualifier in self.result["imports"]:
                return f"{self.result['imports'][qualifier]}.{node.member}"

            # fallback
            return f"{qualifier}.{node.member}"

        return node.member


    def _infer_context(self, node):
        parent = getattr(node, "parent", None)
        if isinstance(parent, javalang.tree.IfStatement):
            return "if-condition"
        if isinstance(parent, javalang.tree.WhileStatement):
            return "while-condition"
        return "expression"
