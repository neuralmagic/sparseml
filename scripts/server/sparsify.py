"""
Sparsify server script. Runs the sparsify server.


##########
Command help:
python scripts/server/sparsify.py -h
usage: sparsify.py [-h] [--working-dir WORKING_DIR] [--host HOST]
                   [--port PORT] [--debug] [--logging-level LOGGING_LEVEL]
                   [--ui-path UI_PATH]

neuralmagicML.server

optional arguments:
  -h, --help            show this help message and exit
  --working-dir WORKING_DIR
                        The path to the working directory to store state in,
                        defaults to ~/nm_server
  --host HOST           The host path to launch the server on
  --port PORT           The local port to launch the server on
  --debug               Set to run in debug mode
  --logging-level LOGGING_LEVEL
                        The logging level to report at
  --ui-path UI_PATH     The directory to render the UI from, generally should
                        not be set. By default, will load from the UI packaged
                        with neuralmagicML under neuralmagicML/server/ui

##########
Example command for running server on a specific host and port
python3 scripts/server/sparsify.py --host 127.0.0.1 --port 3000

Example command for running server and storing defaults in a specific directory
python3 scripts/server/sparsify.py --working-dir ~/Desktop/nm_server
"""

from neuralmagicML.server.app import main

if __name__ == "__main__":
    main()
