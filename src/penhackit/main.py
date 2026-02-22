from penhackit.cli.cli import run_cli


def main():
    """
    Program entrypoint
    """
 
    try:

        # print("Loaded configuration, starting CLI...\n")
        print("1) Configuration")
        print("2) Logging")
        print("3) Environment profiles")
        
        run_cli()


    except KeyboardInterrupt:
        print('\nInterrupted by user. Exiting...')

if __name__ == "__main__":
    main()
