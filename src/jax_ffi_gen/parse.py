import numpy as np

try:
    import tree_sitter_cuda
    from tree_sitter import Language, Parser, Query, QueryCursor, Node, Tree
except ImportError as e:
    raise ImportError("Please install tree-sitter and tree-sitter-cuda packages to use this module.") from e

from dataclasses import dataclass, field

def std_dtype_to_ffi_enum(typename):
    dtype_map = {
        "float": "F32",
        "double": "F64",
        "int": "S32",
        "int32_t": "S32",
        "int64_t": "S64",
        "long long": "S64",
        "int16_t": "S16",
        "short": "S16",
        "int8_t": "S8",
        "uint32_t": "U32",
        "unsigned int": "U32",
        "uint64_t": "U64",
        "unsigned long long": "U64",
        "uint16_t": "U16",
        "unsigned short": "U16",
        "uint8_t": "U8",
        "half": "F16",
        "__half": "F16",
        "bool": "PRED"
    }
    if typename in dtype_map:
        return "DT::" + dtype_map[typename]
    else:
        raise ValueError(f"Unknown datatype {typename}, cannot map to FFI type...")


@dataclass
class ParamInfo():
    type : str = ""
    name : str = ""
    is_ptr : bool = False
    is_const : bool = False
    expression : str = ""
    init_zero : bool = False


@dataclass
class TemplateParamInfo():
    type : str = ""
    name : str = ""
    instances : list[str] = ()
    expression : str = ""
    dtype_from_buffer : str = ""  # Buffer parameter name to extract dtype from

    def ctype(self):
        """Return the C type for dispatch tuples (DT for typename, type otherwise)"""
        return "DT" if self.type == "typename" else self.type

    def dispatch_values(self):
        if self.type != "typename":
            return self.instances
        
        return tuple(map(std_dtype_to_ffi_enum, self.instances))

@dataclass
class FunctionInfo():
    name : str
    par : dict[str, ParamInfo]
    type : str = "void"
    is_kernel : bool = False
    template_par : list[str, ParamInfo] = field(default_factory=dict)
    block_size_expression : str = ""
    grid_size_expression : str = ""
    smem_size_expression : str = ""
    init_outputs_zero : bool = False

    def template_values_flat(self):
        all_perm = np.meshgrid(*[p.instances for p in self.template_par.values()], indexing='ij')
        return np.stack(all_perm, axis=-1).reshape(-1, len(self.template_par))
    
    def dispatch_values_flat(self):
        """Returns dispatch key values derived from instances."""
        values_list = [p.dispatch_values() for p in self.template_par.values()]
        
        all_perm = np.meshgrid(*values_list, indexing='ij')
        return np.stack(all_perm, axis=-1).reshape(-1, len(self.template_par))
    
    def dispatch_values_str(self):
        return [", ".join([str(v) for v in vals]) for vals in self.dispatch_values_flat()]
    
    def template_values_str(self):
        return [", ".join([str(v) for v in vals]) for vals in self.template_values_flat()]

CUDA = Language(tree_sitter_cuda.language())
parser = Parser(CUDA)

def node_text(node: Node, txt: str) -> str:
    return txt[node.start_byte:node.end_byte]

def query(node: Node, query_src: str) -> dict:
    q = QueryCursor(Query(CUDA, query_src))
    caps = q.captures(node)

    return caps

def interprete_parameter_list(node_param: Node, txt: str) -> dict[str, ParamInfo]:
    assert node_param.type == "parameter_list"

    res = {}
    for c in node_param.named_children:
        if c.type == "comment": continue

        assert (c.type == "parameter_declaration") or (c.type == "optional_parameter_declaration")

        pinfo = ParamInfo()

        tq = query(c, '(type_qualifier)? @tq')
        pinfo.is_const = any(node_text(cap, txt) == "const" for cap in tq.get("tq", []))

        pinfo.type = node_text(c.child_by_field_name("type"), txt)

        decl = c.child_by_field_name("declarator")
        if decl.type == "identifier":
            pinfo.name = node_text(decl, txt)
        elif decl.type == "pointer_declarator":
            pinfo.is_ptr = True
            pinfo.name = node_text(decl.child_by_field_name("declarator"), txt)
        else:
            raise ValueError("Unknown type %s" % decl.type)
        
        res[pinfo.name] = pinfo

    return res

def interprete_template_list(node_param: Node, txt: str) -> dict[str, TemplateParamInfo]:
    assert node_param.type == "template_parameter_list"

    res = {}
    for c in node_param.named_children:
        assert c.type in ("parameter_declaration", "type_parameter_declaration", "optional_parameter_declaration")

        pinfo = TemplateParamInfo()
        if c.type == "parameter_declaration":
            pinfo.type = node_text(c.child_by_field_name("type"), txt)
            pinfo.name = node_text(c.child_by_field_name("declarator"), txt)
        elif c.type == "type_parameter_declaration":
            # Handle typename/class template parameters (e.g., "typename T" or "class T")
            pinfo.type = "typename"
            # The type identifier is typically the last named child
            pinfo.name = node_text(c.named_children[-1], txt)
        elif c.type == "optional_parameter_declaration":
            pinfo.type = node_text(c.child_by_field_name("type"), txt)
            pinfo.name = node_text(c.child_by_field_name("declarator"), txt)
        pinfo.instances = []
        
        res[pinfo.name] = pinfo

    return res

def get_functions(node: Node, txt: str, skip_device=True) -> dict[str, FunctionInfo]:
    query_func = f"""
        (function_definition
            ("__device__")? @fdevice
            ("__global__")? @fglobal
            type: (_) @ftype
            declarator: (function_declarator
                declarator: (identifier) @fname
                parameters: (parameter_list) @fparam
            )
        ) @node
    """

    cursor = QueryCursor(Query(CUDA, query_func))
    res = {}
    for i,match in cursor.matches(node):
        if skip_device and (match.get("fdevice") is not None):
            continue

        if (not "ftemp" in match) and () :
            print("template function")
        new_func = FunctionInfo(
            name = node_text(match["fname"][0], txt),
            par = interprete_parameter_list(match["fparam"][0], txt),
            type = node_text(match["ftype"][0], txt),
            is_kernel = match.get("fglobal") is not None
        )
        parent = match["node"][0].parent
        if parent.type == "template_declaration":
            tpar_list = parent.child_by_field_name("parameters")
            new_func.template_par = interprete_template_list(tpar_list, txt)

        res[new_func.name] = new_func
        
    return res

def build_tree_sitter(file_path: str) -> tuple[Tree, str]:
    with open(file_path, 'r') as f:
        txt = f.read()
    tree = parser.parse(txt.encode())
    return tree, txt

def get_functions_from_file(file_path: str, only_kernels: bool = True, names: tuple[str] = None
                            ) -> dict[str, FunctionInfo]:
    tree, txt = build_tree_sitter(file_path)
    funcs = get_functions(tree.root_node, txt)
    
    if only_kernels:
        funcs = {k:v for k,v in funcs.items() if v.is_kernel}

    if names is not None:
        funcs = {k:v for k,v in funcs.items() if k in names}
        for name in names:
            if name not in funcs:
                raise ValueError(f"Function {name} not found in file {file_path}")

    return funcs
