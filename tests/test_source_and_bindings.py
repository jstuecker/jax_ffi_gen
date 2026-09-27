import re

import pytest

from jax_ffi_gen.generator import create_ffi_call
from jax_ffi_gen.parse import FunctionInfo, ParamInfo, get_functions_from_file


@pytest.mark.parametrize("comment", ["ASCII", "∇² café", "🌌"])
def test_source_comments_preserve_function_and_parameter_names(tmp_path, comment):
    source = tmp_path / "kernel.cu"
    source.write_text(
        f"// {comment}\n"
        "template <typename T, int N>\n"
        "__global__ void Scale(\n"
        f"    const T *input, /* {comment} */ T *output, int count) {{}}\n",
        encoding="utf-8",
    )

    function = get_functions_from_file(str(source), names=("Scale",))["Scale"]

    assert function.type == "void"
    assert function.is_kernel
    assert list(function.par) == ["input", "output", "count"]
    assert function.par["input"] == ParamInfo(
        type="T", name="input", is_ptr=True, is_const=True
    )
    assert function.par["output"] == ParamInfo(type="T", name="output", is_ptr=True)
    assert function.par["count"] == ParamInfo(type="int", name="count")
    assert list(function.template_par) == ["T", "N"]
    assert function.template_par["T"].type == "typename"
    assert function.template_par["N"].type == "int"


@pytest.mark.parametrize("is_const", [False, True])
def test_buffer_only_binding_keeps_comma_before_traits(is_const):
    function = FunctionInfo(
        name="Kernel",
        par={"buffer": ParamInfo(
            type="float", name="buffer", is_ptr=True, is_const=is_const
        )},
        is_kernel=True,
        grid_size_expression="1",
        block_size_expression="32",
    )

    code = create_ffi_call(function)
    binding = code.split("XLA_FFI_DEFINE_HANDLER_SYMBOL(", 1)[1]
    binding = re.sub(r"/\*.*?\*/|//[^\n]*", "", binding, flags=re.DOTALL)
    buffer_kind = "Arg" if is_const else "Ret"
    assert re.search(
        rf"\.{buffer_kind}<ffi::AnyBuffer>\(\)\s*,\s*"
        r"\{xla::ffi::Traits::kCmdBufferCompatible\}",
        binding,
    )
