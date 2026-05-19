# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


class _FakeLogger:
    def __init__(self):
        self.info_calls: list[tuple[str, tuple[object, ...]]] = []
        self.info_once_calls: list[tuple[str, tuple[object, ...]]] = []
        self._info_once_keys: set[tuple[str, tuple[object, ...]]] = set()

    def info(self, msg: str, *args: object) -> None:
        self.info_calls.append((msg, args))

    def info_once(self, msg: str, *args: object) -> None:
        key = (msg, args)
        if key in self._info_once_keys:
            return
        self._info_once_keys.add(key)
        self.info_once_calls.append((msg, args))


def _load_npu_freq_module():
    logger_module = ModuleType("vllm.logger")
    logger_module.init_logger = lambda _: _FakeLogger()  # type: ignore[attr-defined]
    vllm_module = ModuleType("vllm")
    vllm_module.__path__ = []  # type: ignore[attr-defined]
    sys.modules.setdefault("vllm", vllm_module)
    sys.modules.setdefault("vllm.logger", logger_module)

    module_path = Path(__file__).parents[2] / "vllm" / "utils" / "npu_freq.py"
    spec = importlib.util.spec_from_file_location("test_npu_freq_module", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


npu_freq = _load_npu_freq_module()


@pytest.fixture(autouse=True)
def reset_npu_freq_state():
    npu_freq._reset_npu_freq_state_for_testing()
    yield
    npu_freq._reset_npu_freq_state_for_testing()


def _fake_torch_npu() -> tuple[ModuleType, list[float]]:
    calls: list[float] = []
    module = ModuleType("torch_npu")
    module.npu_set_freq = calls.append  # type: ignore[attr-defined]
    return module, calls


def test_npu_freq_guard_switches_to_fa_freq_and_back(monkeypatch):
    module, calls = _fake_torch_npu()
    monkeypatch.setitem(sys.modules, "torch_npu", module)

    with npu_freq.npu_fa_freq_guard():
        assert calls == [npu_freq.NPU_FA_FREQ]

    assert calls == [npu_freq.NPU_FA_FREQ, npu_freq.NPU_LOW_FREQ]


def test_npu_freq_skips_redundant_frequency_changes(monkeypatch):
    module, calls = _fake_torch_npu()
    monkeypatch.setitem(sys.modules, "torch_npu", module)

    npu_freq.set_npu_low_freq()
    npu_freq.set_npu_low_freq()
    npu_freq.set_npu_freq(npu_freq.NPU_FA_FREQ)
    npu_freq.set_npu_freq(npu_freq.NPU_FA_FREQ)

    assert calls == [npu_freq.NPU_LOW_FREQ, npu_freq.NPU_FA_FREQ]


def test_npu_freq_logs_actual_frequency_changes(monkeypatch):
    module, calls = _fake_torch_npu()
    logger = _FakeLogger()
    monkeypatch.setitem(sys.modules, "torch_npu", module)
    monkeypatch.setattr(npu_freq, "logger", logger)

    npu_freq.set_npu_low_freq()
    npu_freq.set_npu_low_freq()
    npu_freq.set_npu_freq(npu_freq.NPU_FA_FREQ)

    assert calls == [npu_freq.NPU_LOW_FREQ, npu_freq.NPU_FA_FREQ]
    assert logger.info_calls == [
        ("NPU frequency set: %s -> %s", (None, npu_freq.NPU_LOW_FREQ)),
        (
            "NPU frequency set: %s -> %s",
            (npu_freq.NPU_LOW_FREQ, npu_freq.NPU_FA_FREQ),
        ),
    ]


def test_npu_freq_guard_restores_low_freq_on_exception(monkeypatch):
    module, calls = _fake_torch_npu()
    monkeypatch.setitem(sys.modules, "torch_npu", module)

    with pytest.raises(RuntimeError, match="boom"):
        with npu_freq.npu_fa_freq_guard():
            raise RuntimeError("boom")

    assert calls == [npu_freq.NPU_FA_FREQ, npu_freq.NPU_LOW_FREQ]


def test_npu_freq_is_noop_without_torch_npu(monkeypatch):
    logger = _FakeLogger()
    monkeypatch.setattr(npu_freq, "logger", logger)
    monkeypatch.delitem(sys.modules, "torch_npu", raising=False)

    def raise_import_error(name: str):
        assert name == "torch_npu"
        raise ImportError(name)

    monkeypatch.setattr(npu_freq, "import_module", raise_import_error)

    npu_freq.set_npu_low_freq()
    with npu_freq.npu_fa_freq_guard():
        pass

    assert logger.info_once_calls == [
        ("torch_npu is unavailable; NPU frequency control is disabled.", ())
    ]
