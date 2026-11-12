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
@pytest.mark.parametrize("platform", ["cpu", "cuda"])
def test_buffer_only_binding_keeps_comma_before_traits(is_const, platform):
    function = FunctionInfo(
        name="Kernel",
        par={"buffer": ParamInfo(
            type="float", name="buffer", is_ptr=True, is_const=is_const
        )},
        is_kernel=platform == "cuda",
        platform=platform,
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
def test_requested_functions_skip_unrelated_reference_parameters(tmp_path):
    from jax_ffi_gen.parse import get_functions_from_file

    source = tmp_path / "helpers.cc"
    source.write_text("void Helper(int &count) {}\nvoid Entry(const float *x, float *out) {}")
    functions = get_functions_from_file(str(source), names=["Entry"], only_kernels=False)
    assert list(functions) == ["Entry"]
    assert list(functions["Entry"].par) == ["x", "out"]



@pytest.mark.parametrize("only_kernels,names,expected", [
    (True, None, ["Kernel"]),
    (True, ("Kernel",), ["Kernel"]),
    (False, ("Entry",), ["Entry"]),
])
def test_function_selection_precedes_parameter_parsing(tmp_path, only_kernels, names, expected):
    source = tmp_path / "selection.cu"
    source.write_text(
        "void HostHelper(int values[2]) {}\n"
        "__device__ void DeviceHelper(int values[2]) {}\n"
        "__global__ void Kernel(float *out) {}\n"
        "void Entry(float *out) {}\n"
    )
    functions = get_functions_from_file(str(source), only_kernels=only_kernels, names=names)
    assert list(functions) == expected
    assert list(functions[expected[0]].par) == ["out"]


@pytest.mark.parametrize("parameter", [
    "int &count", "const int &count", "int &&count", "int *&count",
    "int (&count)[2]", "void (&count)(int)",
])
def test_reference_parameters_skip_discovery_but_explain_explicit_requests(tmp_path, parameter):
    source = tmp_path / "references.cc"
    source.write_text(f"void Helper({parameter}) {{}}\nvoid Entry(float *out) {{}}\n")
    functions = get_functions_from_file(str(source), only_kernels=False)
    assert list(functions) == ["Entry"]
    with pytest.raises(ValueError, match=r"Cannot bind function 'Helper'.*count.*reference"):
        get_functions_from_file(str(source), only_kernels=False, names=("Helper",))


def test_missing_requested_function_still_raises(tmp_path):
    source = tmp_path / "entry.cc"
    source.write_text("void Entry(float *out) {}\n")
    with pytest.raises(ValueError, match="Function Missing not found"):
        get_functions_from_file(str(source), only_kernels=False, names=("Missing",))
