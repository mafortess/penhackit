import subprocess


def get_timeout_for_action(action_id: int) -> int:
    if action_id in [211, 220, 230, 401]:
        return 180

    if action_id in [330, 331, 332, 413]:
        return 45

    if 600 <= action_id <= 699:
        return 180

    return 60

def execute_command(cmd):
    stdout_chunks = []
    stderr_chunks = []

    process = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,  # Línea por línea
    )
    
    assert process.stdout is not None
    assert process.stderr is not None

    for line in process.stdout:
        print(line, end="")  # Imprime en tiempo real
        stdout_chunks.append(line) # Guarda en chunks para no perder datos grandes

    for line in process.stderr:
        print(line, end="")  # Imprime en tiempo real
        stderr_chunks.append(line) # Guarda en chunks para no perder datos grandes

    return_code = process.wait()  # Espera a que termine el proceso

    stdout_text = "".join(stdout_chunks)
    stderr_text = "".join(stderr_chunks)

    return return_code, stdout_text, stderr_text

# def execute_command(cmd):
#     print(f"Executing command: {cmd} and capturing result...")
#     if not cmd:  # None o "" => no ejecutar
#         return {"rc": 0, "stdout": "", "stderr": "", "cmd": cmd}
#     o  = subprocess.run(
#         cmd,
#         shell=True,
#         capture_output=True,
#         text=True,
#         timeout=30,
#     )
#     return {
#         "cmd": cmd,
#         "rc": int(o.returncode),
#         "stdout": o.stdout or "",
#         "stderr": o.stderr or "",
#     }
