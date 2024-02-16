#!/usr/bin/env bash
export PATH=/home/flasker/.local/bin:$PATH
cd ..
waitress-serve --port=5022 --ident=vws --url-prefix="/" --call "flask_main:create_app"