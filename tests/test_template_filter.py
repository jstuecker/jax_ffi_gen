from collections import OrderedDict

import pytest

from jax_ffi_gen.generator import create_ffi_call
from jax_ffi_gen.parse import FunctionInfo, TemplateParamInfo


def make_function(template_filter=None):
    return FunctionInfo(
        name="Kernel",
        par={},
        is_kernel=True,
        template_par=OrderedDict(
            p=TemplateParamInfo(type="int", name="p", instances=(1, 2, 3)),
            tvec=TemplateParamInfo(
                type="typename",
                name="tvec",
                instances=("float", "double"),
            ),
        ),
        template_filter=template_filter,
    )


def test_template_filter_receives_keywords_and_preserves_order():
    function = make_function(
        lambda *, p, tvec: p < 3 or (p == 3 and tvec == "float")
    )

    assert function.template_values_flat() == [
        (1, "float"),
        (1, "double"),
        (2, "float"),
        (2, "double"),
        (3, "float"),
    ]
    assert function.dispatch_values_flat() == [
        (1, "DT::F32"),
        (1, "DT::F64"),
        (2, "DT::F32"),
        (2, "DT::F64"),
        (3, "DT::F32"),
    ]


def test_generated_dispatch_only_contains_retained_combinations():
    function = make_function(lambda *, p, tvec: p < 3 or tvec == "float")

    code = create_ffi_call(function)

    assert "&Kernel<3, float>" in code
    assert "&Kernel<3, double>" not in code
    assert "(3, float)" in code
    assert "(3, double)" not in code


def test_template_filter_must_return_bool():
    function = make_function(lambda **_: None)

    with pytest.raises(TypeError, match="must return bool"):
        function.template_values_flat()


def test_template_filter_cannot_remove_every_combination():
    function = make_function(lambda **_: False)

    with pytest.raises(ValueError, match="removed every template combination"):
        function.template_values_flat()


@pytest.mark.parametrize("platform", ["cpu", "cuda"])
def test_host_generation_is_repeatable_and_selects_platform(tmp_path, platform):
    from jax_ffi_gen.parse import get_functions_from_file
    source = tmp_path / 'host.cuh'
    stream = 'cudaStream_t stream, ' if platform == 'cuda' else ''
    source.write_text(f'template<typename T> void Scale({stream}const T *x, T *y, const int n) {{}}')
    fn = get_functions_from_file(str(source), only_kernels=False)['Scale']
    fn.platform = platform
    fn.par['n'].expression = 'x.element_count()'
    fn.template_par['T'].instances = ('float', 'double')
    fn.template_par['T'].expression = 'x.element_type()'
    code = create_ffi_call(fn)
    assert create_ffi_call(fn) == code
    assert ('cudaStream_t' in code) == (platform == 'cuda')
    assert ('cudaGetLastError' in code) == (platform == 'cuda')
    assert '&ScaleDispatchWrapper<float>' in code
    assert '&ScaleDispatchWrapper<double>' in code
    # Trailing comments must not swallow the comma before handler traits.
    assert '.Ret<ffi::AnyBuffer>() /* y */,' in code
