import json
import re
import threading
import time
from pathlib import Path
import matplotlib.pyplot as plt
import requests
import markdown
import subprocess

import torch # para modelos LLM locales (p.ej. llama.cpp con bindings de Python, o modelos más pequeños)
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer # para cargar modelos LLM locales con HuggingFace (si no usas ollama)  

from penhackit.common.paths import Paths, next_available_path

REPORT_SECTIONS = [
    ("Executive Summary", "Resumen ejecutivo: 5-8 líneas, objetivo y resultado general."),
    # ("Scope and Context", "Alcance: objetivo, target(s), entorno (Kali VM + contenedores), restricciones."),
    # ("Environment Observations", "Observaciones del entorno: red local, interfaces, gateways, vecinos ARP relevantes."),
    # ("Actions Performed", "Acciones ejecutadas: lista concisa de comandos y propósito."),
    # ("Findings", "Hallazgos: si no hay, indicar 'No findings in this session' y por qué."),
    ("Next Steps", "Siguientes pasos concretos: 5-10 bullets priorizados."),
]

def generate_report(report_settings: dict, paths: Paths) -> Path:
    kb_path = paths.sessions_dir / report_settings["session_id"] / "kb.json"
    if report_settings["backend"] == "baseline":
        try:
            return generate_report_md_baseline(
                session_dir=paths.sessions_dir / report_settings["session_id"],
                kb=json.loads(kb_path.read_text(encoding="utf-8"))
            )
        except Exception as e:
            print(f"Error generating baseline report: {e}")
            raise
    if report_settings["backend"] == "ollama":
        try:
            return generate_report_md_ollama(
                session_dir=paths.sessions_dir / report_settings["session_id"],
                kb=json.loads(kb_path.read_text(encoding="utf-8")),
                model=report_settings["ollama_model_name"],
            )
        except Exception as e:
            print(f"Error generating report with Ollama: {e}")
            raise
    if report_settings["backend"] == "transformers":
        try:
            return generate_report_md_llm(
                session_dir=paths.sessions_dir / report_settings["session_id"],
                kb=json.loads(kb_path.read_text(encoding="utf-8")),
                backend=report_settings["backend"],
                hf_model_dir= report_settings["transformers_model_name"],
            )
        except Exception as e:
            print(f"Error generating report with Transformers: {e}")
            raise

# MAIN FUNCTION OF THE LLM-BASED REPORT GENERATION (SECTION-WISE, WITH EITHER BACKEND)
def generate_report_md_llm(
    session_dir: Path,
    kb: dict,
    backend: str,
    ollama_model: str | None = None,
    hf_model_dir: Path | None = None,
    hf_device: str = "cpu",
) -> Path:
    print(f"Hf_model_dir: {hf_model_dir})")
    try:
        kb_compact = compact_kb_for_report(kb)
    except Exception as e:
        print(f"Error compacting KB for report: {e}")
        raise
    try:
        report_path = next_available_path(session_dir, "report", ".md")
    except Exception as e:
        print(f"Error determining report path: {e}")
        raise


    header = []
    header.append("# PenHackIt Report")
    header.append("")
    header.append(f"- Session ID: {kb.get('session_id','')}")
    header.append(f"- Generated at: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")

    if backend == "ollama":
        header.append(f"- Backend: ollama")
        header.append(f"- Model: {ollama_model or ''}")
    else:
        header.append(f"- Backend: transformers")
        header.append(f"- Model dir: {str(hf_model_dir) if hf_model_dir else ''}")
        header.append(f"- Device: {hf_device}")

    header.append("")
    report_path.write_text("\n".join(header) + "\n", encoding="utf-8")

    fig_dir = session_dir / "figures"
    plot_counts(kb, fig_dir / "counts.png")
    plot_hosts_kpi(kb, fig_dir / "hosts.png")

    with report_path.open("a", encoding="utf-8") as f:
        f.write("## Figures\n\n")
        f.write("![](figures/counts.png)\n\n")
        f.write("![](figures/hosts.png)\n\n")

    for title, guidance in REPORT_SECTIONS:
        with report_path.open("a", encoding="utf-8") as f:
            f.write(f"## {title}\n\n")

        section_prompt = build_section_prompt(title, guidance, kb_compact)

        if backend == "ollama":
            if not ollama_model:
                raise ValueError("ollama_model is required for backend='ollama'")
            body = ollama_generate_http(ollama_model, section_prompt, timeout_s=180).strip()
        else:
            if not hf_model_dir:
                raise ValueError("hf_model_dir is required for backend='transformers'")
            body = hf_generate_text(
                hf_model_dir,
                section_prompt,
                device=hf_device,
                max_new_tokens=350,
                temperature=0.2,
                top_p=0.95,
            ).strip()
        try:
            body = sanitize_llm_section(body, title)
        except Exception as e:
            print(f"Error sanitizing LLM output for section '{title}': {e}")
            body = f"(Error processing LLM output: {e})"

        try:
            with report_path.open("a", encoding="utf-8") as f:
                f.write(body + "\n\n")
        except Exception as e:
            print(f"Error writing section '{title}' to report: {e}")
            raise

    return report_path

# MAIN FUNCTION OF THE OLLAMA-BASED REPORT GENERATION (SECTION-WISE, WITH EITHER BACKEND)
def generate_report_md_ollama(session_dir: Path, kb: dict, model: str) -> Path:
    kb_compact = compact_kb_for_report(kb)

    # report_path = session_dir / "report.md" # estándar, pero si quieres evitar sobreescribir: next_available_path(session_dir, "report", ".md")
    report_path = next_available_path(session_dir, "report", ".md") # evita sobreescribir

    # 1) cabecera fija PRIMERO (esto crea/sobrescribe el archivo)
    header = []
    header.append("# PenHackIt Report")
    header.append("")
    header.append(f"- Session ID: {kb.get('session_id','')}")
    header.append(f"- Generated at: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    header.append(f"- Model: {model}")
    header.append("")

    report_path.write_text("\n".join(header) + "\n", encoding="utf-8")

    # 2) generar figuras y AÑADIRLAS (append) DESPUÉS
    fig_dir = session_dir / "figures"

    plot_counts(kb, fig_dir / "counts.png")
    plot_hosts_kpi(kb, fig_dir / "hosts.png")

    with (report_path).open("a", encoding="utf-8") as f:
        f.write("## Figures\n\n")
        f.write("![](figures/counts.png)\n\n")
        f.write("![](figures/hosts.png)\n\n")

    # 3) secciones fijas (append)
    for title, guidance in REPORT_SECTIONS:
        with report_path.open("a", encoding="utf-8") as f:
            f.write(f"## {title}\n\n")

        prompt = build_section_prompt(title, guidance, kb_compact)
        body = ollama_generate_http(model, prompt, timeout_s=180).strip()
        body = sanitize_llm_section(body, title)

        with report_path.open("a", encoding="utf-8") as f:
            f.write(body + "\n\n")

    return report_path

# MAIN FUNCTION OF THE BASELINE (NO LLM) REPORT GENERATION
def generate_report_md_baseline(session_dir: Path, kb: dict) -> Path:
    print("Generating baseline report (no LLM)...")
    try:
        kb_compact = compact_kb_for_report(kb)
    except Exception as e:
        print(f"Error compacting KB for baseline report: {e}")
        raise
    
    try:
        report_path = next_available_path(session_dir, "report", ".md")
    except Exception as e:
        print(f"Error determining report path for baseline report: {e}")
        raise
    
    header = []
    header.append("# PenHackIt Report")
    header.append("")
    header.append(f"- Session ID: {kb.get('session_id','')}")
    header.append(f"- Generated at: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    header.append(f"- Backend: baseline (no LLM)")
    header.append("")

    report_path.write_text("\n".join(header) + "\n", encoding="utf-8")

    fig_dir = session_dir / "figures"
    plot_counts(kb, fig_dir / "counts.png")
    plot_hosts_kpi(kb, fig_dir / "hosts.png")

    with report_path.open("a", encoding="utf-8") as f:
        f.write("## Figures\n\n")
        f.write("![](figures/counts.png)\n\n")
        f.write("![](figures/hosts.png)\n\n")

    for title, guidance in REPORT_SECTIONS:
        with report_path.open("a", encoding="utf-8") as f:
            f.write(f"## {title}\n\n")
        body = baseline_section_body(title, kb_compact).strip()
        with report_path.open("a", encoding="utf-8") as f:
            f.write(body + "\n\n")

    return report_path


# FUNCIONES AUXILIARES PARA GENERACIÓN DE REPORTES (LLM Y BASELINE)

def baseline_section_body(section_title: str, kb_compact: dict) -> str:
    """
    Baseline deterministic generator: no LLM.
    Uses only kb_compact, does not invent facts.
    """
    counts = kb_compact.get("counts", {}) or {}
    net = kb_compact.get("net", {}) or {}
    focus = kb_compact.get("focus", {}) or {}
    sid = kb_compact.get("session_id") or ""
    goal_type = kb_compact.get("goal_type") or "unknown"
    target = kb_compact.get("target") or "unknown"

    hosts_n = int(counts.get("hosts", 0) or 0)
    services_n = int(counts.get("services", 0) or 0)
    findings_n = int(counts.get("findings", 0) or 0)
    notes_n = int(counts.get("notes", 0) or 0)
    commands_n = int(counts.get("commands", 0) or 0)

    ipv4 = net.get("ipv4", []) or []
    gws = net.get("default_gw", []) or []
    arps = net.get("arp_neighbors", []) or []

    commands_tail = kb_compact.get("commands_tail", []) or []
    findings_sample = kb_compact.get("findings_sample", []) or []
    hosts_sample = kb_compact.get("hosts_sample", []) or []
    services_sample = kb_compact.get("services_sample", []) or []
    notes_tail = kb_compact.get("notes_tail", []) or []

    def bullet_list(items: list[str], empty_msg: str = "No data captured in KB.") -> str:
        if not items:
            return empty_msg
        return "\n".join([f"- {x}" for x in items])

    def fmt_arp(n: dict) -> str:
        ip = (n or {}).get("ip", "") or ""
        mac = (n or {}).get("mac", "") or ""
        kind = (n or {}).get("type", "") or ""
        s = ip
        if mac:
            s += f" ({mac})"
        if kind:
            s += f" [{kind}]"
        return s.strip() or "(unknown)"

    def fmt_host(h: dict) -> str:
        ip = (h or {}).get("ip", "") or ""
        src = (h or {}).get("source", "") or ""
        return f"{ip} [{src}]" if src else ip

    def fmt_service(svc: dict) -> str:
        # si tu KB no tiene estructura de services aún, esto no inventa nada
        if not isinstance(svc, dict):
            return str(svc)
        parts = []
        for k in ("host", "ip", "port", "proto", "name", "product", "version"):
            v = svc.get(k)
            if v is None or v == "":
                continue
            parts.append(f"{k}={v}")
        return ", ".join(parts) if parts else str(svc)

    if section_title == "Executive Summary":
        lines = []
        lines.append(f"Session {sid} executed as a baseline report (no LLM).")
        lines.append(f"Goal: {goal_type}. Target: {target}.")
        lines.append(f"Captured: {commands_n} commands, {hosts_n} hosts, {services_n} services, {findings_n} findings, {notes_n} notes.")
        if findings_n == 0:
            lines.append("Outcome: no findings recorded in KB for this session.")
        else:
            lines.append("Outcome: findings were recorded in KB; see Findings section for details.")
        return " ".join(lines)

    if section_title == "Scope and Context":
        lines = []
        lines.append(f"Scope is limited to the data captured in the session KB and command outputs.")
        lines.append(f"Goal type: {goal_type}. Target: {target}.")
        fl = (focus or {}).get("level", "global")
        fh = (focus or {}).get("host", "") or ""
        fs = (focus or {}).get("service", "") or ""
        lines.append(f"Focus: level={fl}, host={fh or 'none'}, service={fs or 'none'}.")
        lines.append("Environment details (OS, tooling versions, constraints) are not fully captured unless stored in KB.")
        return "\n".join(lines)

    if section_title == "Environment Observations":
        lines = []
        lines.append("Network observations captured from KB:")
        lines.append("")
        lines.append(f"- Local IPv4(s): {', '.join(ipv4) if ipv4 else 'Not captured'}")
        lines.append(f"- Default gateway(s): {', '.join(gws) if gws else 'Not captured'}")
        if arps:
            lines.append(f"- ARP neighbors ({min(len(arps), 30)} shown):")
            for n in arps[:30]:
                lines.append(f"  - {fmt_arp(n)}")
        else:
            lines.append("- ARP neighbors: Not captured")
        return "\n".join(lines)

    if section_title == "Actions Performed":
        # lista solo lo que existe en KB, sin interpretación
        if not commands_tail:
            return "No commands recorded in KB."
        items = [c for c in commands_tail if isinstance(c, str) and c.strip()]
        return bullet_list(items, empty_msg="No commands recorded in KB.")

    if section_title == "Findings":
        if findings_n == 0:
            return "No findings in this session (KB.findings is empty)."
        # si findings son dicts, los serializamos de forma segura
        out = []
        for f in findings_sample[:30]:
            if isinstance(f, dict):
                out.append(json.dumps(f, ensure_ascii=False))
            else:
                out.append(str(f))
        return bullet_list(out, empty_msg="Findings present but not readable.")

    if section_title == "Next Steps":
        steps = []
        # heurísticas simples basadas en "qué falta"
        if hosts_n == 0:
            steps.append("Capture discovery output to populate KB.hosts (e.g., ARP/scan results).")
        elif services_n == 0:
            steps.append("Capture service enumeration to populate KB.services (ports/protocols/banners).")
        if commands_n == 0:
            steps.append("Ensure executed commands are appended to KB.commands for traceability.")
        if findings_n == 0:
            steps.append("If the goal is vulnerability assessment, add steps that produce findings and store them in KB.findings.")
        if not ipv4 and not gws:
            steps.append("Capture basic network context (IPv4/default gateway/interfaces) to support environment section.")
        if not steps:
            steps.append("Refine KB schema for the target scenario and increase coverage of actions/parsers.")
            steps.append("Run additional sessions to build a larger dataset for BC training and evaluation.")
        return bullet_list(steps)

    # fallback (shouldn't happen)
    return "No baseline implementation for this section."



def compact_kb_for_report(kb: dict) -> dict:
    net = kb.get("net", {}) or {}
    return {
        "session_id": kb.get("session_id"),
        "goal_type": (kb.get("session_context", {}) or {}).get("goal_type"),  # si existe
        "target": (kb.get("session_context", {}) or {}).get("target"),        # si existe
        "focus": kb.get("focus", {}),
        "counts": {
            "hosts": len(kb.get("hosts", []) or []),
            "services": len(kb.get("services", []) or []),
            "findings": len(kb.get("findings", []) or []),
            "notes": len(kb.get("notes", []) or []),
            "commands": len(kb.get("commands", []) or []),
        },
        "net": {
            "ipv4": (net.get("ipv4", []) or [])[:10],
            "default_gw": (net.get("default_gw", []) or [])[:10],
            "arp_neighbors": (net.get("arp_neighbors", []) or [])[:30],
        },
        "hosts_sample": (kb.get("hosts", []) or [])[:30],
        "services_sample": (kb.get("services", []) or [])[:30],
        "findings_sample": (kb.get("findings", []) or [])[:30],
        "commands_tail": (kb.get("commands", []) or [])[-40:],
        "notes_tail": (kb.get("notes", []) or [])[-20:],
    }

def build_section_prompt(section_title: str, section_guidance: str, kb_compact: dict) -> str:
    return f"""You are writing ONLY the BODY for a pentest report section.

Section title: {section_title}
Guidance: {section_guidance}

Rules:
- Do NOT include any headings or the section title.
- Be concise and technical.
- Do not invent facts. Use only the KB.
- If there is insufficient data, explicitly say so and suggest what data is missing.

KB (JSON):
{json.dumps(kb_compact, ensure_ascii=False, indent=2)}
"""


OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"

def ollama_generate_http(model: str, prompt: str, timeout_s: int = 180) -> str:
    r = requests.post(
        OLLAMA_GENERATE_URL,
        json={"model": model, "prompt": prompt, "stream": True},
        stream=True,
        timeout=timeout_s,
    )
    r.raise_for_status()

    chunks = []
    for line in r.iter_lines(decode_unicode=True):
        if not line:
            continue
        data = json.loads(line)
        token = data.get("response", "")
        if token:
            print(token, end="", flush=True)  # feedback inmediato
            chunks.append(token)
        if data.get("done"):
            break

    print()  # newline final
    return "".join(chunks).strip()

def sanitize_llm_section(text: str, section_title: str) -> str:
    t = (text or "").strip()

    # 1) remove triple-backtick fences (keep inner text)
    # removes ```lang\n ... \n``` blocks
    t = re.sub(r"```[a-zA-Z0-9_-]*\n", "", t)
    t = t.replace("```", "")

    # 2) remove accidental repeated headings at start (e.g. "## Executive Summary")
    lines = [ln.rstrip() for ln in t.splitlines()]
    while lines and re.match(r"^#{1,6}\s+", lines[0]):
        # if it's the same section title or any heading, drop it
        h = re.sub(r"^#{1,6}\s+", "", lines[0]).strip().lower()
        if h == section_title.strip().lower():
            lines.pop(0)
            # drop following blank lines
            while lines and lines[0].strip() == "":
                lines.pop(0)
        else:
            break

    t = "\n".join(lines).strip()
    return t


# 5) Transformers generator with caching (load once, reuse per section)
_HF_CACHE = {
    "model_dir": None,
    "device": None,
    "tokenizer": None,
    "model": None,
}

def hf_load_model(model_dir: Path, device: str):
    global _HF_CACHE
    if _HF_CACHE["model_dir"] == str(model_dir) and _HF_CACHE["device"] == device and _HF_CACHE["model"] is not None:
        return _HF_CACHE["tokenizer"], _HF_CACHE["model"]

    # Clean previous (optional)
    _HF_CACHE = {"model_dir": str(model_dir), "device": device, "tokenizer": None, "model": None}

    print(f"\n[HF] Loading model from: {model_dir}")
    tok = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)

    # If tokenizer has no pad_token (common), set it to eos to avoid warnings in generation
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

    # For small models: float16 on cuda; float32 on cpu
    dtype = torch.float16 if device == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )

    if device == "cuda":
        model = model.to("cuda")
    else:
        model = model.to("cpu")

    model.eval()

    _HF_CACHE["tokenizer"] = tok
    _HF_CACHE["model"] = model
    return tok, model

def hf_generate_stream(model_dir: Path, prompt_text: str, max_new_tokens: int = 350) -> str:
    print(f"[HF] Loading model from: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        local_files_only=True,
        device_map="auto",
        torch_dtype="auto",
    )
    model.eval()

    inputs = tokenizer(prompt_text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    gen_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    t = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    t.start()

    chunks = []
    for tok in streamer:
        print(tok, end="", flush=True)
        chunks.append(tok)

    print()
    t.join()
    return "".join(chunks).strip()

def hf_generate_text(
    model_dir: Path,
    prompt_text: str,
    device: str,
    max_new_tokens: int = 350,
    temperature: float = 0.2,
    top_p: float = 0.95,
) -> str:
    print(f"\n[HF] Loading model from: {model_dir} on device: {device}")
    tok, model = hf_load_model(model_dir, device)

    inputs = tok(prompt_text, return_tensors="pt")
    if device == "cuda":
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    else:
        inputs = {k: v.to("cpu") for k, v in inputs.items()}

    # Stream output tokens to console (similar feel to your Ollama streaming)
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)

    gen_kwargs = dict(
        **inputs,
        max_new_tokens=int(max_new_tokens),
        do_sample=True if temperature > 0 else False,
        temperature=float(temperature),
        top_p=float(top_p),
        streamer=streamer,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )

    # Run generation in a background thread so streamer can iterate

    t = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    t.start()

    chunks = []
    for new_text in streamer:
        print(new_text, end="", flush=True)
        chunks.append(new_text)
    print()
    t.join()

    return "".join(chunks).strip()


# =================================================================
# UTILIDADES VARIAS PARA LOS REPORTES (FIGURAS, PDF, ETC)

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm

def md_to_pdf_simple(md_path: Path, pdf_path: Path) -> None:
    text = md_path.read_text(encoding="utf-8").splitlines()

    c = canvas.Canvas(str(pdf_path), pagesize=A4)
    width, height = A4

    x = 2 * cm
    y = height - 2 * cm
    line_h = 14  # puntos

    def new_page():
        nonlocal y
        c.showPage()
        y = height - 2 * cm

    for line in text:
        # estilo básico por heurística
        if line.startswith("# "):
            c.setFont("Helvetica-Bold", 16)
            line = line[2:].strip()
            y -= 6
        elif line.startswith("## "):
            c.setFont("Helvetica-Bold", 13)
            line = line[3:].strip()
            y -= 4
        elif line.startswith("- "):
            c.setFont("Helvetica", 10)
            line = "• " + line[2:].strip()
        else:
            c.setFont("Helvetica", 10)

        # salto de página si hace falta
        if y < 2 * cm:
            new_page()

        # dibuja línea
        c.drawString(x, y, line[:120])  # recorte simple MVP
        y -= line_h

    c.save()

def md_to_pdf(md_path: Path, pdf_path: Path) -> None:
    md_text = md_path.read_text(encoding="utf-8")
    html_body = markdown.markdown(md_text, extensions=["fenced_code", "tables"])
    html = f"""
    <html>
      <head>
        <meta charset="utf-8">
        <style>
          body {{ font-family: Arial, sans-serif; font-size: 12px; }}
          h1, h2 {{ margin-top: 20px; }}
          pre {{ background: #f5f5f5; padding: 10px; overflow-x: auto; }}
          code {{ font-family: Consolas, monospace; }}
        </style>
      </head>
      <body>{html_body}</body>
    </html>
    """
    HTML(string=html).write_pdf(str(pdf_path))

# FIGURE GENERATION (USADO POR TODOS LOS BACKENDS, INDEPENDIENTE DE LLM)
def plot_hosts_kpi(kb: dict, out_png: Path) -> None:
    n = len(kb.get("hosts", []) or [])
    out_png.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.axis("off")
    plt.text(0.5, 0.6, "Hosts discovered", ha="center", va="center", fontsize=18)
    plt.text(0.5, 0.35, str(n), ha="center", va="center", fontsize=48)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()

def plot_counts(kb: dict, out_png: Path) -> None:
    hosts = len(kb.get("hosts", []) or [])
    services = len(kb.get("services", []) or [])
    findings = len(kb.get("findings", []) or [])
    commands = len(kb.get("commands", []) or [])

    labels = ["hosts", "services", "findings", "commands"]
    values = [hosts, services, findings, commands]

    # Asegura que el directorio existe para guardar la figura
    out_png.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.bar(labels, values)
    plt.ylabel("count")
    plt.title("Session summary counts")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

