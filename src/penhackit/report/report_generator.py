import json
from pathlib import Path

from penhackit.report.report_schema import REPORT_SECTIONS
from penhackit.report.report_llm_writer import LLMWriter


def generate_report_md(session_id: str, llm: LLMWriter, sessions_dir: str = "data/sessions") -> str:
    """
    Minimal flow validator:
      - reads data/sessions/<session_id>/kb.json
      - iterates REPORT_SECTIONS
      - writes data/sessions/<session_id>/report.md
    """
    session_root = Path(sessions_dir) / session_id
    kb_path = session_root / "kb.json"
    if not kb_path.is_file():
        raise FileNotFoundError(f"Missing KB: {kb_path}")

    kb = json.loads(kb_path.read_text(encoding="utf-8"))

    out_lines = []
    for section_id, title, level in REPORT_SECTIONS:
        out_lines.append(f"{'#' * int(level)} {title}\n\n")

        prompt = (
            "Write the body text for this report section. Do NOT include the title.\n"
            f"SECTION_ID: {section_id}\n"
            f"SECTION_TITLE: {title}\n"
        )
        text = (llm.generate(prompt, kb) or "").strip()
        out_lines.append(text + "\n\n")

    out_path = session_root / "report.md"
    out_path.write_text("".join(out_lines), encoding="utf-8")
    return str(out_path)