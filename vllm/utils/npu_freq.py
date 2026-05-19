# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Generator
from contextlib import contextmanager
from importlib import import_module
from types import ModuleType

from vllm.logger import init_logger

NPU_LOW_FREQ = 400.0
NPU_FA_FREQ = 1650.0

logger = init_logger(__name__)

_torch_npu: ModuleType | None = None
_torch_npu_import_attempted = False
_current_freq: float | None = None


def _get_torch_npu() -> ModuleType | None:
    global _torch_npu_import_attempted, _torch_npu

    if _torch_npu_import_attempted:
        return _torch_npu

    _torch_npu_import_attempted = True
    try:
        _torch_npu = import_module("torch_npu")
    except Exception:
        _torch_npu = None
    return _torch_npu


def set_npu_freq(freq: float) -> None:
    global _current_freq

    if _current_freq == freq:
        return

    old_freq = _current_freq
    torch_npu = _get_torch_npu()
    if torch_npu is None:
        logger.info_once("torch_npu is unavailable; NPU frequency control is disabled.")
        return

    torch_npu.npu_set_freq(freq)
    _current_freq = freq
    logger.info("NPU frequency set: %s -> %s", old_freq, freq)


def set_npu_low_freq() -> None:
    set_npu_freq(NPU_LOW_FREQ)


@contextmanager
def npu_fa_freq_guard() -> Generator[None, None, None]:
    set_npu_freq(NPU_FA_FREQ)
    try:
        yield
    finally:
        set_npu_low_freq()


def _reset_npu_freq_state_for_testing() -> None:
    global _current_freq, _torch_npu, _torch_npu_import_attempted

    _current_freq = None
    _torch_npu = None
    _torch_npu_import_attempted = False
