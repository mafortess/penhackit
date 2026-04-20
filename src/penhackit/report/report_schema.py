# Estructura fija mínima para validar el flujo.
# Cada item: (section_id, title, md_level)
# REPORT_SECTIONS = [
#     ("summary", "Summary", 1),
#     ("findings", "Findings", 2),
#     ("notes", "Notes", 2),
# ]

REPORT_SECTIONS = [
    ("Executive Summary", "Resumen ejecutivo: 5-8 líneas, objetivo y resultado general."),
    # ("Scope and Context", "Alcance: objetivo, target(s), entorno (Kali VM + contenedores), restricciones."),
    # ("Environment Observations", "Observaciones del entorno: red local, interfaces, gateways, vecinos ARP relevantes."),
    # ("Actions Performed", "Acciones ejecutadas: lista concisa de comandos y propósito."),
    # ("Findings", "Hallazgos: si no hay, indicar 'No findings in this session' y por qué."),
    ("Next Steps", "Siguientes pasos concretos: 5-10 bullets priorizados."),
]

REPORT_TEMPLATES = {
    "minimal": [
        ("Executive Summary", "Resumen ejecutivo breve(3-4 líneas)."),
        ("Next Steps", "Siguientes pasos. Máximo 3 acciones concretas."),
    ],

    "standard": [
        ("Executive Summary", "Resumen ejecutivo: 5-8 líneas, objetivo y resultado general."),
        ("Scope and Context", "Alcance: objetivo, target(s), entorno (Kali VM + contenedores), restricciones."),
        ("Environment Observations", "Observaciones del entorno: red local, interfaces, gateways, vecinos ARP relevantes."),
        ("Actions Performed", "Acciones ejecutadas: lista concisa de comandos y propósito."),
        ("Findings", "Hallazgos: si no hay, indicar 'No findings in this session' y por qué."),
        ("Next Steps", "Siguientes pasos concretos: 5-10 bullets priorizados.")
    ],

    "technical": [
        ("Executive Summary", "Resumen ejecutivo."),
        ("Environment Observations", "Observaciones del entorno."),
        ("Actions Performed", "Acciones ejecutadas."),
        ("Findings", "Hallazgos técnicos."),
        ("Next Steps", "Siguientes pasos."),
    ],
}
