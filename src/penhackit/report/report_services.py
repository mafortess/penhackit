from penhackit.report.report_wizard import generate_report_wizard
from penhackit.report.report_logic import generate_report, md_to_pdf_simple

from penhackit.common.paths import next_available_path, Paths

def run_generate_report_service(app_context: dict) -> None:
    print(f"Running report generation service...") # with context: {app_context}")

    # Load default settings and paths
    report_settings = app_context["settings"]["report"]
    print(f"Default report settings: {report_settings}")
    paths = app_context["paths"]

    # Wizard for report generation
    wizard_data = generate_report_wizard(report_settings, paths)
    if wizard_data is None:
        print("Report generation cancelled.")
        return
   
    report_settings = {
        "session_id": wizard_data["session_id"],
        "output_format": wizard_data["output_format"],
        "backend": wizard_data["backend"],
        "ollama_model_name": wizard_data["ollama_model_name"],
        "transformers_model_name": wizard_data["transformers_model_name"],
        # "pdf_generation": wizard_data["pdf_generation"],
        "device": wizard_data["device"],
    }

    # report_settings = {
    #     "session_id": "session_test",
    #     "output_format": report_settings["default_output_format"],  # e.g. "md", "html", "pdf"
    #     "backend": report_settings["default_backend"],  # or "ollama", "transformers"
    #     "ollama_model_name": report_settings["default_ollama_model"],  # e.g. "gpt-3.5-turbo" for ollama, or local model name for transformers
    #     "transformers_model_name": report_settings["default_transformers_model"],  # e.g. "gpt2" for transformers, or local model name for transformers
    #     "pdf_generation": report_settings["export_pdf_after_generation"],
    # }
    try:
        print("Generating report with the following settings:")
        for key, value in report_settings.items():
            print(f"  {key}: {value}")

        md_path = generate_report(report_settings, paths)
        print("Report generated successfully in Markdown format.")
        print(f"Markdown report saved to: {md_path}")
        if report_settings["output_format"] == "md+pdf":
            print("PDF export is enabled, the report should be available in PDF format.")
            print("Introduce the name for the PDF report (without extension):")
            name_report = input("> ").strip()
            pdf_path = next_available_path(paths.sessions_dir / f"{report_settings['session_id']}", f"{name_report}", "pdf")
            try:
                md_to_pdf_simple(md_path, pdf_path)
                print(f"PDF report saved to: {pdf_path}")
            except Exception as e:
                print(f"Error during PDF generation: {e}")

    except Exception as e:
        print(f"Error during report generation: {e}")



def list_reports() -> None:
    print("Listing reports...")
    # Aquí iría la lógica para listar los informes existentes, mostrando información relevante.

def show_report_details(report_id: str) -> None:
    print(f"Showing details for report {report_id}...")
    # Aquí iría la lógica para mostrar los detalles de un informe específico, probablemente pidiendo un ID o nombre.


def wizard_generate_report(app_context: dict) -> None:
    print("Starting report generation wizard...")
    # Aquí iría la lógica para guiar al usuario a través de los pasos necesarios para generar un informe, probablemente usando funciones del módulo wizard.py para cada paso.