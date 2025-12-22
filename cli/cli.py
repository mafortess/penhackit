import os
import sys
import yaml
from pathlib import Path
# from capture.terminal_logger import TerminalCapturer
# # from preprocessing.preprocess_to_json import convert_all
# from preprocessing.preprocess import preprocess_session_files
# from training.train_bc__ import train_bc
# from actuator.actuator import Actuator
# from actuator.actuator_autonomo import Actuator as ActuatorAutonomo

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / 'config' / 'config.yaml'

def ensure_dirs(cfg):
    for p in [cfg['paths']['logs_dir'], cfg['paths']['captures_dir'], cfg['paths']['training_logs'], cfg['paths']['models_dir']]:
        os.makedirs(p, exist_ok=True)

def load_config():
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)

def menu():
    cfg = load_config()
    ensure_dirs(cfg)    


    while True:
        print('\n=== AGENT COMMAND-LINE INTERFACE (CLI) ===')
        print('1) Start recorder (data capture modules)')
        print('2) Preprocess sessions')
        print('3) Train BC model (baseline)')
        print('4) Run agent (suggestion mode)')
        print('5) Run agent (autonomous mode)')
        print('0) Exit')


        c = input('> ')
        if c == '1':
            print('Iniciando grabadora...')
            tc = TerminalCapturer(cfg)
            tc.record_interactive()

        elif c == '2':
            print('Preprocesando sesiones...')
            preprocess_session_files(cfg)

        elif c == '3':
            print('[*] Entrenando modelo Behavioral Cloning (scikit-learn)...')
            # train_bc(cfg)
            train_bc(
                captures_path=cfg['paths']['captures_dir'].replace("captures", "sessions"),
                output_dir=cfg['paths']['models_dir']
            )
            # train_bc(
            #     captures_path=cfg['paths']['captures_dir'],
            #     output_dir=cfg['paths']['models_dir']
            # )

        elif c == '4':
            print('[*] Iniciando actuador (modo sugerencia)...')
            act = Actuator(cfg)
            act.suggestion_loop()

        elif c == '5':
            print('[*] Iniciando actuador (modo autónomo)...')
            act = ActuatorAutonomo(cfg)
            act.autonomous_loop()
            
        elif c == '0':
            print('Saliendo')
            sys.exit(0)
        else:
            print('Opción no válida')

if __name__ == '__main__':
    menu()