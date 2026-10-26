"""
WebGPU toolchain: megakernel GLSL -> SPIR-V (glslang) -> lowered SPIR-V (see
_spirv_lower) -> WGSL (Tint).

External tools are looked up, in order, from an explicit environment
variable, the PATH, and $VULKAN_SDK/Bin:

    glslangValidator  RDV_GLSLANG
    spirv-dis/as/val  RDV_SPIRV_TOOLS (a directory holding all three)
    tint              RDV_TINT        (from a Dawn build/release)
"""

import os as _os
import shutil as _shutil
import subprocess as _subprocess
import tempfile as _tempfile

from ._spirv_lower import lower_spirv, SpirvLoweringError


class WebGPUExportError(RuntimeError):
    pass


def _find_tool(name, env_var=None, env_is_dir=False):
    exe = name + ('.exe' if _os.name == 'nt' else '')
    explicit = _os.environ.get(env_var) if env_var else None
    if explicit:
        path = _os.path.join(explicit, exe) if env_is_dir else explicit
        if _os.path.isfile(path):
            return path
        raise WebGPUExportError(f'{env_var}={explicit!r} does not point to {exe}.')
    found = _shutil.which(name)
    if found:
        return found
    sdk = _os.environ.get('VULKAN_SDK')
    if sdk:
        for sub in ('Bin', 'bin'):
            path = _os.path.join(sdk, sub, exe)
            if _os.path.isfile(path):
                return path
    hint = f' (or set {env_var})' if env_var else ''
    raise WebGPUExportError(f'{name} was not found on PATH or in $VULKAN_SDK{hint}.')


def _run(args, what, source=None):
    proc = _subprocess.run(args, capture_output=True, text=True)
    if proc.returncode != 0:
        message = f'{what} failed:\n{proc.stdout}\n{proc.stderr}'.strip()
        if source is not None:
            numbered = '\n'.join(f'{i + 1}\t{line}' for i, line in enumerate(source.split('\n')))
            message += f'\n\n--- source ---\n{numbered}'
        raise WebGPUExportError(message)
    return proc.stdout


def glsl_to_wgsl(code: str, include_dirs=(), keep_dir: str | None = None, split_heap: bool = True,
                 dynamic_base: int | None = None):
    """
    Compiles a WebGPU-flavoured megakernel (see _build_code(..., webgpu=True))
    all the way to WGSL. Returns (wgsl source, bindings, warnings) - see
    lower_spirv for bindings. keep_dir, if given, receives every
    intermediate file (megakernel.comp/.spv/.spvasm, lowered.spvasm/.spv),
    which is the first thing to look at when Tint rejects something.
    split_heap/dynamic_base: see _spirv_lower (loads from a read-only heap,
    stores to a read-write mirror; False keeps a single read-write heap).
    """
    glslang = _find_tool('glslangValidator', 'RDV_GLSLANG')
    spirv_dis = _find_tool('spirv-dis', 'RDV_SPIRV_TOOLS', env_is_dir=True)
    spirv_as = _find_tool('spirv-as', 'RDV_SPIRV_TOOLS', env_is_dir=True)
    spirv_val = _find_tool('spirv-val', 'RDV_SPIRV_TOOLS', env_is_dir=True)
    tint = _find_tool('tint', 'RDV_TINT')

    work = keep_dir or _tempfile.mkdtemp(prefix='rdv_webgpu_')
    _os.makedirs(work, exist_ok=True)
    try:
        def path(name):
            return _os.path.join(work, name)

        with open(path('megakernel.comp'), 'w') as f:
            f.write(code)
        # vulkan1.1 -> SPIR-V 1.3, the newest version Tint's reader accepts.
        _run([glslang, '-S', 'comp', '--target-env', 'vulkan1.1', *[f'-I{d}' for d in include_dirs],
              '-o', path('megakernel.spv'), path('megakernel.comp')], 'glslang', source=code)
        _run([spirv_dis, '--raw-id', '-o', path('megakernel.spvasm'), path('megakernel.spv')], 'spirv-dis')
        with open(path('megakernel.spvasm')) as f:
            asm = f.read()
        try:
            lowered, bindings, warnings = lower_spirv(asm, split_heap=split_heap, dynamic_base=dynamic_base)
        except SpirvLoweringError as e:
            raise WebGPUExportError(f'SPIR-V lowering failed: {e}') from e
        with open(path('lowered.spvasm'), 'w') as f:
            f.write(lowered)
        _run([spirv_as, '--target-env', 'spv1.3', '-o', path('lowered.spv'), path('lowered.spvasm')], 'spirv-as')
        _run([spirv_val, '--target-env', 'vulkan1.1', path('lowered.spv')], 'spirv-val (lowered module)')
        _run([tint, '--format', 'wgsl', '-o', path('megakernel.wgsl'), path('lowered.spv')], 'tint')
        with open(path('megakernel.wgsl')) as f:
            wgsl = f.read()
        return wgsl, bindings, warnings
    finally:
        if keep_dir is None:
            _shutil.rmtree(work, ignore_errors=True)
