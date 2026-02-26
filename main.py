import psutil
import sys
import requests
import re
import platform
from bs4 import BeautifulSoup
from huggingface_hub import HfApi

from textual.app import App, ComposeResult
from textual.widgets import Input, DataTable, Footer, Static, RadioSet, RadioButton, Button, Label
from textual.screen import ModalScreen
from textual.containers import Grid, Vertical, Horizontal
from textual import work, on

# --- YARDIMCI FONKSİYONLAR ---
def get_real_cpu_name():
    os_name = platform.system()
    if os_name == "Windows":
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            cpu_name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            return cpu_name.strip()
        except: return platform.processor()
    elif os_name == "Linux":
        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line: return line.split(":")[1].strip()
        except: return platform.processor()
    elif os_name == "Darwin":
        try:
            import subprocess
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).strip().decode("utf-8")
        except: return platform.processor()
    return platform.processor()

class HardwareMonitor:
    def __init__(self):
        self.nvidia_available = False
        self.handle = None
        self.gpu_name = "NVIDIA GPU Yok"
        self.cpu_name = get_real_cpu_name()
        self.cpu_cores = psutil.cpu_count(logical=True)
        
        try:
            import nvidia_smi
            nvidia_smi.nvmlInit()
            self.handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
            self.gpu_name = nvidia_smi.nvmlDeviceGetName(self.handle)
            if isinstance(self.gpu_name, bytes): self.gpu_name = self.gpu_name.decode('utf-8')
            self.nvidia_available = True
        except:
            try:
                import pynvml
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.gpu_name = pynvml.nvmlDeviceGetName(self.handle)
                if isinstance(self.gpu_name, bytes): self.gpu_name = self.gpu_name.decode('utf-8')
                self.nvidia_available = True
            except: pass

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
            "has_gpu": self.nvidia_available
        }
        
        if self.nvidia_available:
            try:
                try: import nvidia_smi as nv
                except: import pynvml as nv
                mem = nv.nvmlDeviceGetMemoryInfo(self.handle)
                info["vram_free"] = mem.free / (1024**3)
                info["vram_total"] = mem.total / (1024**3)
            except: pass
        return info

def check_ollama_running():
    for proc in psutil.process_iter(['name']):
        try:
            if proc.info['name'] and 'ollama' in proc.info['name'].lower(): return True
        except: pass
    return False

def get_installed_ollama_models():
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=1)
        if response.status_code == 200:
            return [m["name"].split(":")[0].lower() for m in response.json().get("models",[])]
    except: return[]
    return[]

def extract_params(name):
    match = re.search(r'(\d+(?:\.\d+)?[BbMm])', name, re.IGNORECASE)
    return match.group(1).upper() if match else "-"

def format_likes(num):
    if num >= 1000000: return f"{num/1000000:.1f}M"
    if num >= 1000: return f"{num/1000:.1f}K"
    return str(num)

def determine_use_case(name):
    name_lower = name.lower()
    if any(kw in name_lower for kw in["coder", "code", "starcoder", "deepseek-coder"]): return "[bold blue]Coding[/bold blue]"
    elif any(kw in name_lower for kw in["vision", "vl", "llava", "pixtral"]): return "[bold magenta]Vision[/bold magenta]"
    elif any(kw in name_lower for kw in ["math"]): return "[bold cyan]Math[/bold cyan]"
    elif any(kw in name_lower for kw in["reasoning", "think", "-r1", "deepseek-r1"]): return "[bold yellow]Reasoning[/bold yellow]"
    elif any(kw in name_lower for kw in ["embed", "bge", "nomic"]): return "[grey74]Embedding[/grey74]"
    elif any(kw in name_lower for kw in["instruct", "chat", "dolphin", "hermes"]): return "[bold green]Chat[/bold green]"
    else: return "[white]General[/white]"

def calculate_fit(size_gb, specs):
    buffer = 1.5 
    req = size_gb + buffer
    if specs['has_gpu'] and specs['vram_free'] >= req: return "[bold green]Perfect[/bold green]", "[green]GPU[/green]", "VRAM"
    elif specs['has_gpu'] and (specs['vram_free'] + specs['ram_free']) >= req: return "[bold yellow]Partial[/bold yellow]", "[yellow]GPU+CPU[/yellow]", "VRAM+RAM"
    elif specs['ram_free'] >= req: return "[bold yellow]Slow[/bold yellow]", "[yellow]CPU[/yellow]", "RAM"
    else: return "[bold red]No Fit[/bold red]", "[red]-[/red]", "Yetersiz"

# --- MODAL (POPUP) EKRANI ---
class ModelDetailModal(ModalScreen):
    """Satıra tıklanınca açılan detay penceresi"""
    CSS = """
    ModelDetailModal {
        align: center middle;
        background: rgba(0,0,0,0.7);
    }
    #modal-container {
        width: 60%;
        height: auto;
        background: $surface;
        border: round $accent;
        padding: 1 2;
    }
    #modal-title {
        text-align: center;
        text-style: bold;
        background: $primary;
        color: white;
        padding: 1;
        margin-bottom: 1;
    }
    #info-grid {
        layout: grid;
        grid-size: 2;
        grid-gutter: 1;
        height: auto;
    }
    .label-key { 
        color: $text-muted; 
        text-style: bold; 
        margin-top: 1; /* Buraya taşındı, hata çözüldü */
    }
    #cmd-box {
        background: $panel;
        border: solid green;
        padding: 1;
        margin: 1 0;
        text-align: center;
        color: $accent-lighten-2;
    }
    #close-btn { width: 100%; margin-top: 1; }
    """

    def __init__(self, data):
        super().__init__()
        self.data = data

    def compose(self) -> ComposeResult:
        cmd_text = ""
        if self.data['source'] == "Ollama":
            cmd_text = f"ollama run {self.data['name']}"
        else:
            cmd_text = f"huggingface-cli download {self.data['name']} --include '*.gguf'"

        with Vertical(id="modal-container"):
            yield Label(f"{self.data['name']}", id="modal-title")
            
            with Grid(id="info-grid"):
                yield Label(f"[bold]Sağlayıcı:[/bold] {self.data['provider']}")
                yield Label(f"[bold]Kullanım:[/bold] {self.data['use_case']}")
                yield Label(f"[bold]Parametre:[/bold] {self.data['params']}")
                yield Label(f"[bold]Format:[/bold] {self.data['quant']}")
                yield Label(f"[bold]Skor:[/bold] {self.data['score']}")
                yield Label(f"[bold]Tahmini Boyut:[/bold] {self.data['size']}")
                yield Label(f"[bold]Donanım Durumu:[/bold] {self.data['fit']}")
                yield Label(f"[bold]Çalışma Modu:[/bold] {self.data['mode']}")

            yield Label("Çalıştırma / İndirme Komutu:", classes="label-key")
            yield Label(cmd_text, id="cmd-box")
            
            yield Button("Kapat", variant="error", id="close-btn")

    @on(Button.Pressed, "#close-btn")
    def close_modal(self):
        self.dismiss()

# --- ANA UYGULAMA ---
class SystemInfoWidget(Static):
    def update_info(self, specs, ollama_running):
        gpu_color = "green" if specs['has_gpu'] else "red"
        ollama_status = "[bold green]✔ Açık[/bold green]" if ollama_running else "[bold red]X Kapalı[/bold red]"
        text = (
            f"[bold]CPU:[/bold] {specs['cpu_name']} ({specs['cpu_cores']} cores) | "
            f"[bold]RAM:[/bold][cyan]{specs['ram_free']:.1f} GB Boş[/cyan] / {specs['ram_total']:.1f} GB | "
            f"[bold]GPU:[/bold] [{gpu_color}]{specs['gpu_name']} ({specs['vram_free']:.1f} GB Boş)[/{gpu_color}] | "
            f"[bold]Ollama:[/bold] {ollama_status}"
        )
        self.update(text)

class AIModelViewer(App):
    CSS = """
    Screen { layout: vertical; padding: 1; }
    SystemInfoWidget { height: 3; border: round cyan; content-align: center middle; margin-bottom: 1; }
    Input { width: 100%; margin-bottom: 1; }
    RadioSet { layout: horizontal; width: 100%; height: 3; border: none; align: center middle; margin-bottom: 1; }
    DataTable { height: 1fr; border: round grey; }
    """

    BINDINGS =[("q", "quit", "Çıkış"), ("tab", "focus_next", "Sonraki"), ("shift+tab", "focus_previous", "Önceki")]

    def __init__(self):
        super().__init__()
        self.monitor = HardwareMonitor()
        self.all_results = []
        self.current_filter = "All"

    def compose(self) -> ComposeResult:
        yield SystemInfoWidget(id="header")
        yield Input(placeholder="🔍 Model Ara (Örn: llama, qwen) ve Enter'a bas...", id="search-input")
        with RadioSet(id="filter-set"):
            yield RadioButton("Tümü", value=True, id="filter-all")
            yield RadioButton("Ollama", id="filter-ollama")
            yield RadioButton("Hugging Face", id="filter-hf")
        yield DataTable(id="results-table", cursor_type="row")
        yield Footer()

    def on_mount(self) -> None:
        self.title = "AI Model Explorer"
        table = self.query_one(DataTable)
        table.add_columns("Yüklü", "Kaynak", "Model Adı", "Param", "Kullanım", "Skor", "Format", "Çalışma", "Durum", "Boyut")
        self.update_system_info()
        self.set_interval(3.0, self.update_system_info)

    def update_system_info(self):
        specs = self.monitor.get_specs()
        ollama_running = check_ollama_running()
        self.query_one(SystemInfoWidget).update_info(specs, ollama_running)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        query = event.value.strip()
        if not query: return
        table = self.query_one(DataTable)
        table.clear()
        table.loading = True
        self.run_search_worker(query)

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        pid = event.pressed.id
        if pid == "filter-all": self.current_filter = "All"
        elif pid == "filter-ollama": self.current_filter = "Ollama"
        elif pid == "filter-hf": self.current_filter = "Hugging Face"
        self.refresh_table()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        # DataTable'dan gelen row_key'i string olarak alıyoruz
        row_key_str = str(event.row_key.value)
        # Unique key (Source:Name) üzerinden eşleşen modeli bul
        selected_model = next((m for m in self.all_results if f"{m['source']}:{m['name']}" == row_key_str), None)
        
        if selected_model:
            self.push_screen(ModelDetailModal(selected_model))

    @work(thread=True)
    def run_search_worker(self, query: str) -> None:
        specs = self.monitor.get_specs()
        local_models = get_installed_ollama_models() if check_ollama_running() else []
        results = []
        found_keys = set() # Duplicate engelleme

        # OLLAMA
        try:
            url = f"https://ollama.com/search?q={query}"
            headers = {"User-Agent": "Mozilla/5.0"}
            resp = requests.get(url, headers=headers, timeout=5)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, 'html.parser')
                for a_tag in soup.find_all('a', href=True):
                    href = a_tag['href']
                    if href.startswith("/library/") and "/blog/" not in href and "/tags" not in href:
                        model_name = href.replace("/library/", "").strip()
                        unique_k = f"Ollama:{model_name}"
                        if unique_k in found_keys: continue
                        found_keys.add(unique_k)
                        
                        full_text = a_tag.get_text(strip=True)
                        pulls = re.search(r'(\d+(?:\.\d+)?[KM]?)\s*Pulls', full_text, re.IGNORECASE)
                        if not pulls:
                            parent = a_tag.find_parent('li')
                            if parent: pulls = re.search(r'(\d+(?:\.\d+)?[KM]?)\s*Pulls', parent.get_text(strip=True), re.IGNORECASE)

                        score_str = f"[cyan]📥 {pulls.group(1)}[/cyan]" if pulls else "[grey50]-[/grey50]"
                        params = extract_params(model_name)
                        inst = "[green]✔[/green]" if model_name.lower() in local_models else "[grey37]-[/grey37]"
                        use_case = determine_use_case(model_name)
                        
                        es = 4.8
                        mn = model_name.lower()
                        if "70b" in mn or "72b" in mn: es = 40.0
                        elif "32b" in mn: es = 19.0
                        elif "27b" in mn: es = 16.0
                        elif "14b" in mn or "13b" in mn: es = 8.0
                        elif "8b" in mn or "7b" in mn: es = 4.8
                        elif "1.5b" in mn: es = 1.2
                        elif "0.5b" in mn: es = 0.6
                        
                        fit_str, mode_str, _ = calculate_fit(es, specs)
                        results.append({
                            "inst": inst, "source": "Ollama", "provider": "Ollama Registry",
                            "name": model_name, "params": params, "use_case": use_case, "score": score_str, "quant": "Q4_0", 
                            "mode": mode_str, "fit": fit_str, "size": f"~{es} GB"
                        })
        except: pass

        # HF
        try:
            api = HfApi()
            hf_models = api.list_models(search=query, sort="downloads", limit=15, filter="gguf")
            for model in hf_models:
                provider = model.modelId.split("/")[0][:15]
                name = model.modelId.split("/")[-1]
                unique_k = f"Hugging Face:{name}"
                if unique_k in found_keys: continue
                found_keys.add(unique_k)

                params = extract_params(name)
                use_case = determine_use_case(name)
                likes = getattr(model, 'likes', 0)
                score_str = f"[red]❤️ {format_likes(likes)}[/red]" if likes > 0 else "[grey50]-[/grey50]"
                
                files = api.list_repo_files(repo_id=model.modelId)
                target = next((f for f in files if "Q4_K_M.gguf" in f), None)
                if not target: target = next((f for f in files if "Q4_0.gguf" in f), None)
                if not target: target = next((f for f in files if "Q5_K_M.gguf" in f), None)
                if not target: target = next((f for f in files if f.endswith(".gguf")), None)
                
                if target:
                    info = api.model_info(model.modelId, files_metadata=True)
                    f_meta = next((s for s in info.siblings if s.rfilename == target), None)
                    if f_meta:
                        size = f_meta.size / (1024**3)
                        fit_str, mode_str, _ = calculate_fit(size, specs)
                        quant = target.split('.')[-2] if len(target.split('.')) > 2 else "GGUF"
                        if "gguf" in quant.lower(): quant = "GGUF"
                        results.append({
                            "inst": "[grey37]-[/grey37]", "source": "Hugging Face", "provider": provider,
                            "name": name, "params": params, "use_case": use_case, "score": score_str, "quant": quant, 
                            "mode": mode_str, "fit": fit_str, "size": f"{size:.1f} GB"
                        })
        except: pass

        self.all_results = results
        self.call_from_thread(self.on_search_completed)

    def on_search_completed(self) -> None:
        table = self.query_one(DataTable)
        table.loading = False
        self.refresh_table()
        table.focus()

    def refresh_table(self):
        table = self.query_one(DataTable)
        table.clear()
        
        added_in_table = set()
        for r in self.all_results:
            if self.current_filter == "All" or r["source"] == self.current_filter:
                disp_name = r["name"]
                if len(disp_name) > 30:
                    disp_name = disp_name[:27] + "..."
                
                unique_key = f"{r['source']}:{r['name']}"
                if unique_key in added_in_table: continue
                added_in_table.add(unique_key)
                
                table.add_row(
                    r["inst"], r["source"], disp_name, r["params"], 
                    r["use_case"], r["score"], r["quant"], r["mode"], 
                    r["fit"], r["size"],
                    key=unique_key
                )

if __name__ == "__main__":
    app = AIModelViewer()
    app.run()