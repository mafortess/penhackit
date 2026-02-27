from penhackit.app.startup import bootstrap_app
from penhackit.cli.main import run_cli

def main():
    try:
        app_context = bootstrap_app()
        run_cli(app_context)

    except KeyboardInterrupt:
        print('\nInterrupted by user. Exiting...')