from frontend import *
from backend import *
from streamlit import runtime

def main():
    display(ingest, query_and_augment)

if __name__ == "__main__":
    if runtime.exists():
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
