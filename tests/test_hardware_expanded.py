"""Tests for expanded hardware detection — AMD, Intel, Apple, backend."""

from hardware import (
    HardwareMonitor,
    detect_gpu_vendor,
    detect_gpu_vendor_from_name,
    get_backend_label,
)


class TestDetectGpuVendor:
    def test_nvidia_returns_cuda(self):
        monitor = HardwareMonitor.__new__(HardwareMonitor)
        monitor.nvidia_available = True
        monitor.gpu_name = "NVIDIA GeForce RTX 4090"
        vendor = detect_gpu_vendor(monitor)
        assert vendor == "nvidia"

    def test_amd_returns_rocm(self):
        vendor = detect_gpu_vendor_from_name("AMD Radeon RX 7900 XTX")
        assert vendor == "amd"

    def test_intel_returns_intel(self):
        vendor = detect_gpu_vendor_from_name("Intel Arc A770")
        assert vendor == "intel"

    def test_apple_returns_metal(self):
        vendor = detect_gpu_vendor_from_name("Apple M2 Ultra")
        assert vendor == "apple"

    def test_unknown_returns_none(self):
        vendor = detect_gpu_vendor_from_name("Unknown GPU")
        assert vendor is None


class TestGetBackendLabel:
    def test_nvidia_is_cuda(self):
        assert get_backend_label("nvidia") == "cuda"

    def test_amd_is_rocm(self):
        assert get_backend_label("amd") == "rocm"

    def test_apple_is_metal(self):
        assert get_backend_label("apple") == "metal"

    def test_intel_is_sycl(self):
        assert get_backend_label("intel") == "sycl"

    def test_unknown_is_cpu(self):
        assert get_backend_label(None) == "cpu"

    def test_empty_is_cpu(self):
        assert get_backend_label("") == "cpu"


class TestHardwareMonitorSpecs:
    def test_specs_has_backend_field(self):
        monitor = HardwareMonitor()
        specs = monitor.get_specs()
        assert "backend" in specs
        assert specs["backend"] in ("cuda", "rocm", "metal", "sycl", "cpu")

    def test_specs_has_gpu_vendor_field(self):
        monitor = HardwareMonitor()
        specs = monitor.get_specs()
        assert "gpu_vendor" in specs

    def test_specs_has_multi_gpu_field(self):
        monitor = HardwareMonitor()
        specs = monitor.get_specs()
        assert "gpu_count" in specs
        assert isinstance(specs["gpu_count"], int)
        assert specs["gpu_count"] >= 0

    def test_specs_has_total_vram_field(self):
        monitor = HardwareMonitor()
        specs = monitor.get_specs()
        assert "vram_total_all" in specs
        assert specs["vram_total_all"] >= 0
