class LLMWriter:
    """
    Interfaz mínima para el generador.
    Implementa generate(prompt, kb) -> str
    """
    def generate(self, prompt: str, kb: dict) -> str:
        raise NotImplementedError


class DummyLLMWriter(LLMWriter):
    """
    Para validar el flujo sin LLM real.
    """
    def generate(self, prompt: str, kb: dict) -> str:
        # Contenido mínimo y estable (sin fallos).
        sid = kb.get("session_metadata", {}).get("session_id", "unknown")
        return f"(dummy) Generated text for session {sid}."