import platform
import subprocess

import psutil


def get_real_cpu_name():
    """Return the human-readable CPU model string for the current platform.

    Uses platform-appropriate APIs (Windows registry, ``/proc/cpuinfo``, macOS
    ``sysctl``) and falls back to :func:`platform.processor` on failure.
    """
    os_name = platform.system()
    if os_name == "Windows":
        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            )
            cpu_name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            return cpu_name.strip()
        except OSError:
            return platform.processor()
    if os_name == "Linux":
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as file:
                for line in file:
                    if "model name" in line:
                        return line.split(":", maxsplit=1)[1].strip()
        except (OSError, FileNotFoundError):
            return platform.processor()
    if os_name == "Darwin":
        try:
            return (
                subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"])
                .strip()
                .decode("utf-8")
            )
        except (OSError, FileNotFoundError, subprocess.SubprocessError):
            return platform.processor()
    return platform.processor()


# ---------------------------------------------------------------------------
# GPU Vendor Detection
# ---------------------------------------------------------------------------


def detect_gpu_vendor_from_name(gpu_name: str) -> str | None:
    """Detect GPU vendor from a GPU name string.

    Returns: ``"nvidia"``, ``"amd"``, ``"intel"``, ``"apple"``, or ``None``.
    """
    if not gpu_name:
        return None
    lower = gpu_name.lower()
    if "nvidia" in lower or "geforce" in lower or "rtx" in lower or "gtx" in lower:
        return "nvidia"
    if "amd" in lower or "radeon" in lower or "rx " in lower or "mi2" in lower or "mi3" in lower:
        return "amd"
    if "intel" in lower or "arc " in lower:
        return "intel"
    if "apple" in lower or "m1" in lower or "m2" in lower or "m3" in lower or "m4" in lower:
        return "apple"
    return None


def detect_gpu_vendor(monitor: "HardwareMonitor") -> str | None:
    """Detect GPU vendor from a HardwareMonitor instance."""
    if monitor.nvidia_available:
        return "nvidia"
    if getattr(monitor, "amd_available", False):
        return "amd"
    if getattr(monitor, "intel_available", False):
        return "intel"
    if getattr(monitor, "apple_available", False):
        return "apple"
    return None


def get_backend_label(vendor: str | None) -> str:
    """Map GPU vendor to compute backend label.

    Returns: ``"cuda"``, ``"rocm"``, ``"metal"``, ``"sycl"``, or ``"cpu"``.
    """
    return {
        "nvidia": "cuda",
        "amd": "rocm",
        "apple": "metal",
        "intel": "sycl",
    }.get(vendor or "", "cpu")


class HardwareMonitor:
    """Snapshot local hardware capabilities relevant to LLM inference.

    Detects NVIDIA (via pynvml/nvidia_smi), AMD (via rocm-smi),
    Apple Silicon (via system_profiler), and Intel Arc (via sysfs/lspci)
    GPUs with graceful fallback when no compatible GPU is present.
    """

    def __init__(self):
        self.nvidia_available = False
        self.amd_available = False
        self.intel_available = False
        self.apple_available = False
        self.handle = None
        self.gpu_name = "No GPU detected"
        self.gpu_count = 0
        self.cpu_name = get_real_cpu_name()
        self.cpu_cores = psutil.cpu_count(logical=True)

        # Try NVIDIA first (most common for LLM)
        self._detect_nvidia()

        # Try other vendors if NVIDIA not found
        if not self.nvidia_available:
            self._detect_amd()
        if not self.nvidia_available and not self.amd_available:
            self._detect_apple()
        if not self.nvidia_available and not self.amd_available and not self.apple_available:
            self._detect_intel()

    def _detect_nvidia(self):
        """Detect NVIDIA GPU via pynvml or nvidia_smi."""
        try:
            import nvidia_smi

            nvidia_smi.nvmlInit()
            self.handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
            self.gpu_name = nvidia_smi.nvmlDeviceGetName(self.handle)
            if isinstance(self.gpu_name, bytes):
                self.gpu_name = self.gpu_name.decode("utf-8")
            self.nvidia_available = True
            self.gpu_count = nvidia_smi.nvmlDeviceGetCount()
            return
        except (ImportError, OSError, RuntimeError):
            pass

        try:
            import pynvml

            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.gpu_name = pynvml.nvmlDeviceGetName(self.handle)
            if isinstance(self.gpu_name, bytes):
                self.gpu_name = self.gpu_name.decode("utf-8")
            self.nvidia_available = True
            self.gpu_count = pynvml.nvmlDeviceGetCount()
        except (ImportError, OSError, RuntimeError):
            pass

    def _detect_amd(self):
        """Detect AMD GPU via rocm-smi."""
        try:
            result = subprocess.run(
                ["rocm-smi", "--showproductname"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                for line in result.stdout.strip().splitlines():
                    if "card" in line.lower() or "gpu" in line.lower():
                        # Extract GPU name after ":"
                        parts = line.split(":", maxsplit=1)
                        if len(parts) > 1:
                            self.gpu_name = f"AMD {parts[1].strip()}"
                        else:
                            self.gpu_name = "AMD GPU"
                        self.amd_available = True
                        self.gpu_count = 1
                        # Try to get VRAM
                        self._detect_amd_vram()
                        return
        except (FileNotFoundError, subprocess.SubprocessError, subprocess.TimeoutExpired):
            pass

    def _detect_amd_vram(self):
        """Attempt to read AMD VRAM info (stored for get_specs)."""
        try:
            result = subprocess.run(
                ["rocm-smi", "--showmeminfo", "vram"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                self._amd_vram_raw = result.stdout
            else:
                self._amd_vram_raw = None
        except (FileNotFoundError, subprocess.SubprocessError, subprocess.TimeoutExpired):
            self._amd_vram_raw = None

    def _detect_apple(self):
        """Detect Apple Silicon unified memory via system_profiler."""
        if platform.system() != "Darwin":
            return
        try:
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                output = result.stdout.lower()
                if (
                    "apple" in output
                    or "m1" in output
                    or "m2" in output
                    or "m3" in output
                    or "m4" in output
                ):
                    # Extract chip name
                    for line in result.stdout.splitlines():
                        if "chip" in line.lower() or "apple" in line.lower():
                            self.gpu_name = line.strip().split(":")[-1].strip()
                            if not self.gpu_name.startswith("Apple"):
                                self.gpu_name = f"Apple {self.gpu_name}"
                            break
                    self.apple_available = True
                    self.gpu_count = 1
                    self._apple_memory_gb = self._get_apple_memory()
        except (FileNotFoundError, subprocess.SubprocessError, subprocess.TimeoutExpired):
            pass

    def _get_apple_memory(self) -> float:
        """Get unified memory size on Apple Silicon."""
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                bytes_val = int(result.stdout.strip())
                return bytes_val / (1024**3)
        except (ValueError, subprocess.SubprocessError, subprocess.TimeoutExpired):
            pass
        return psutil.virtual_memory().total / (1024**3)

    def _detect_intel(self):
        """Detect Intel Arc GPU via lspci or sysfs."""
        try:
            result = subprocess.run(
                ["lspci"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    if "vga" in line.lower() and "intel" in line.lower() and "arc" in line.lower():
                        self.gpu_name = line.split(":", maxsplit=1)[-1].strip()
                        self.intel_available = True
                        self.gpu_count = 1
                        return
        except (FileNotFoundError, subprocess.SubprocessError, subprocess.TimeoutExpired):
            pass

    def get_specs(self):
        """Return a dict with current CPU, RAM, and VRAM readings.

        Keys: ``cpu_name``, ``cpu_cores``, ``ram_free``, ``ram_total``,
        ``vram_free``, ``vram_total``, ``gpu_name``, ``has_gpu``,
        ``gpu_vendor``, ``backend``, ``gpu_count``, ``vram_total_all``.
        All memory values are in **gigabytes**.
        """
        ram = psutil.virtual_memory()
        vendor = detect_gpu_vendor(self)
        backend = get_backend_label(vendor)

        info = {
            "cpu_name": self.cpu_name,
            "cpu_cores": self.cpu_cores,
            "ram_free": ram.available / (1024**3),
            "ram_total": ram.total / (1024**3),
            "vram_free": 0.0,
            "vram_total": 0.0,
            "gpu_name": self.gpu_name,
            "has_gpu": self.nvidia_available
            or self.amd_available
            or self.apple_available
            or self.intel_available,
            "gpu_vendor": vendor,
            "backend": backend,
            "gpu_count": self.gpu_count,
            "vram_total_all": 0.0,
        }

        if self.nvidia_available:
            try:
                try:
                    import nvidia_smi as nv
                except ImportError:
                    import pynvml as nv
                mem = nv.nvmlDeviceGetMemoryInfo(self.handle)
                info["vram_free"] = mem.free / (1024**3)
                info["vram_total"] = mem.total / (1024**3)
                info["vram_total_all"] = info["vram_total"]
            except (ImportError, OSError, RuntimeError, AttributeError):
                pass

        elif self.apple_available:
            # Apple Silicon: unified memory = VRAM
            unified_mem = getattr(self, "_apple_memory_gb", ram.total / (1024**3))
            info["vram_free"] = ram.available / (1024**3)
            info["vram_total"] = unified_mem
            info["vram_total_all"] = unified_mem

        elif self.amd_available:
            # Try to parse AMD VRAM from rocm-smi output
            raw = getattr(self, "_amd_vram_raw", None)
            if raw:
                import re

                total_match = re.search(
                    r"vram.*?total.*?(\d+(?:\.\d+)?)\s*(?:MB|GB)", raw, re.IGNORECASE
                )
                free_match = re.search(
                    r"vram.*?used.*?(\d+(?:\.\d+)?)\s*(?:MB|GB)", raw, re.IGNORECASE
                )
                if total_match:
                    total_val = float(total_match.group(1))
                    if "mb" in total_match.group(0).lower():
                        total_val /= 1024
                    info["vram_total"] = total_val
                    info["vram_total_all"] = total_val
                if free_match:
                    used_val = float(free_match.group(1))
                    if "mb" in free_match.group(0).lower():
                        used_val /= 1024
                    info["vram_free"] = max(0, info["vram_total"] - used_val)

        elif self.intel_available:
            # Intel Arc — VRAM detection limited, use system RAM as approximation
            info["vram_total"] = 0.0
            info["vram_total_all"] = 0.0

        return info


def check_ollama_running():
    """Return ``True`` if an Ollama process is currently running on this machine."""
    for proc in psutil.process_iter(["name"]):
        try:
            if proc.info["name"] and "ollama" in proc.info["name"].lower():
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return False
