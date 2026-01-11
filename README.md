# JAX-FFI-GEN

A few scripts that help with auto-generating jax's foreign function interface (FFI) binding for CUDA kernels
This code uses `tree_sitter` to parse CUDA code and `jinja2` to auto-generate the corresponding FFI. This is useful for establishing a workflow where one "almost" directly calls CUDA kernels rather than worrying too much about the large amount of boiler plate code that comes with jax's FFI.

It is recommended to put a little python script next to the CUDA source files and execute it every time one needs to regenerate some source file, because the corresponding kernel interface changed.

```python
from pathlib import Path
from jax_ffi_gen import parse, generator

HERE = Path(__file__).resolve().parent

kernels = parse.get_functions_from_file(str(HERE / "my_kernels.cuh"), only_kernels=True)

generator.generate_ffi_module_file(
    output_file = str(HERE / "generated/ffi_my_kernels.cu"), 
    functions = kernels
)
```

The parser will interprete each argument of your kernel as follows:
* `const *` pointer type as an input jax array
* `*` modifiable pointer type as an output jax array
* `const` as a static parameter that needs to be known at jit-compile time

Some useful customization options
```python
kernels = parse.get_functions_from_file(
    str(HERE / "my_kernels.cuh"), 
    names = ["MyKernelA", "MyKernelB"] # only select some kernels by name
)

# Examples of a few useful features that you may need to define per kenrel
kernels["MyKernelA"].init_outputs_zero = True
kernels["MyKernelA"].grid_size_expression = "x.element_count()"
kernels["MyKernelA"].block_size_expression = "64"
kernels["MyKernelA"].smem_size_expression = "blockDim.x * sizeof(float4)" # dynamic shared memory
kernels["MyKernelA"].par["num_particles"].expression = "x.element_count()/3"
kernels["MyKernelA"].template_par["p"].instances = (0,1,2)

generator.generate_ffi_module_file(
    output_file = str(HERE / "generated/ffi_new_kernels.cu"), 
    functions = kernels,
    includes = ["../math.cuh"] # set includes
)
```