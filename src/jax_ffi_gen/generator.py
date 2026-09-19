from .parse import FunctionInfo
from jinja2 import Environment, PackageLoader, select_autoescape
from dataclasses import dataclass, replace
import os

env = Environment(
    loader=PackageLoader("jax_ffi_gen", "templates"),
    autoescape=select_autoescape()
)

def simplify_and_validate(func: FunctionInfo) -> FunctionInfo:
    if func.platform not in ("cuda", "cpu"):
        raise ValueError(f"Unknown platform: {func.platform}")
    if func.platform == "cpu" and func.is_kernel:
        raise ValueError("CPU generation requires a host function")
    if not func.is_kernel and func.platform == "cuda":
        assert tuple(func.par.keys())[0] == "stream", "All Host functions must use stream as first parameter"
        func = replace(func, par={k: v for k, v in func.par.items() if k != "stream"})

    # convert pars to lists for easier templating
    func = replace(
        func, 
        par = list(func.par.values())
    )

    for name,p in func.template_par.items():
        if (len(p.instances) == 0) and (p.type == "bool"):
            p.instances = ("true", "false")
        elif len(p.instances) == 0:
            raise ValueError(f"Please define instances for template parameter {p.name} "
                            f"in function {func.name}.")

    return func

def create_ffi_call(func: FunctionInfo) -> str:
    func = simplify_and_validate(func)

    template = env.get_template("template_ffi_call.j2")
    return template.render(f=func).strip() + "\n"

def create_ffi_module_code(funcs: list[FunctionInfo], 
                           includes: tuple[str] = (), 
                           module_name: str = "ffi_module") -> str:
    if type(funcs) is dict:
        funcs = list(funcs.values())

    new_funcs = []
    for f in funcs:
        new_funcs.append(simplify_and_validate(f))

    template = env.get_template("template_ffi_module.j2")
    return template.render(functions=new_funcs, includes=includes, module_name=module_name)

def generate_ffi_module_file(output_file: str, 
                             functions: list[FunctionInfo],
                             includes: tuple[str] = (),
                             module_name: str | None = None) -> None:
    if module_name is None:
        module_name = output_file.split("/")[-1].split(".")[0]

    code = create_ffi_module_code(functions, includes, module_name)

    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            txt = f.read()
        if txt == code:
            print(f"No changes to generated file at {output_file}")
            return
        else:
            print(f"Updating generated file at {output_file}")
    else: 
        print(f"Generating new file file at {output_file}")
    
    with open(output_file, 'w') as f:
        f.write(code)

def create_ffi_registration_code(functions: list[tuple[FunctionInfo, str]],
                                 platform_guards: dict[str, str] | None = None) -> str:
    """Export compiled handlers as {platform: {target_name: capsule}}.

    Include this header in a nanobind module and export ``FFIRegistrations``.
    A platform guard excludes both declarations and references to optional
    handlers, allowing CPU-only builds without CUDA headers or libraries.
    Target names may be shared across platforms, but must be unique within one.
    """
    platform_guards = platform_guards or {}
    grouped = {}
    for fn, target in functions:
        if fn.platform not in ('cpu', 'cuda'):
            raise ValueError(f'Unknown platform: {fn.platform}')
        entries = grouped.setdefault(fn.platform, {})
        if target in entries:
            raise ValueError(f'Duplicate FFI target {target!r} for {fn.platform}')
        entries[target] = fn.name

    def guarded(platform, lines):
        guard = platform_guards.get(platform)
        return ([f'#ifdef {guard}'] + lines + ['#endif']) if guard else lines

    import json
    lines = ['// Generated FFI registration; do not edit.',
             '#include <nanobind/nanobind.h>', '#include "xla/ffi/api/ffi.h"', '']
    for platform, entries in grouped.items():
        lines += guarded(platform, [f'XLA_FFI_DECLARE_HANDLER_SYMBOL({name}FFI);'
                                    for name in entries.values()])
    lines += ['', 'inline nanobind::dict FFIRegistrations() {', '    nanobind::dict result;']
    for platform, entries in grouped.items():
        body = ['    {', '        nanobind::dict targets;']
        for target, name in entries.items():
            body += [f'        targets[{json.dumps(target)}] = nanobind::capsule(',
                     f'            reinterpret_cast<void *>(&{name}FFI), "xla._CUSTOM_CALL_TARGET");']
        body += [f'        result["{ "CUDA" if platform == "cuda" else "cpu" }"] = targets;', '    }']
        lines += guarded(platform, body)
    return '\n'.join(lines + ['    return result;', '}', ''])
