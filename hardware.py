import platform
import subprocess

import psutil


def get_real_cpu_name():
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
            with open("/proc/cpuinfo", "r", encoding="utf-8") as file:
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


class HardwareMonitor:
    def __init__(self):
        self.nvidia_available = False
        self.handle = None
        self.gpu_name = "No NVIDIA GPU"
        self.cpu_name = get_real_cpu_name()
        self.cpu_cores = psutil.cpu_count(logical=True)

        try:
            import nvidia_smi

            nvidia_smi.nvmlInit()
            self.handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
            self.gpu_name = nvidia_smi.nvmlDeviceGetName(self.handle)
            if isinstance(self.gpu_name, bytes):
                self.gpu_name = self.gpu_name.decode("utf-8")
            self.nvidia_available = True
        except (ImportError, OSError, RuntimeError):
            try:
                import pynvml

                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.gpu_name = pynvml.nvmlDeviceGetName(self.handle)
                if isinstance(self.gpu_name, bytes):
                    self.gpu_name = self.gpu_name.decode("utf-8")
                self.nvidia_available = True
            except (ImportError, OSError, RuntimeError):
                pass

    def get_specs(self):
        ram = psutil.virtual_memory()
        info = {
            "cpu_name": self.cpu_name,
            "cpu_cores": self.cpu_cores,
            "ram_free": ram.available / (1024**3),
            "ram_total": ram.total / (1024**3),
            "vram_free": 0.0,
            "vram_total": 0.0,
            "gpu_name": self.gpu_name,
            "has_gpu": self.nvidia_available,
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
            except (ImportError, OSError, RuntimeError, AttributeError):
                pass
        return info


def check_ollama_running():
    for proc in psutil.process_iter(["name"]):
        try:
            if proc.info["name"] and "ollama" in proc.info["name"].lower():
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return False
