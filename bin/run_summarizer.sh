#!/usr/bin/env bash
#gunicorn -w 3 --timeout 2000 -b 0.0.0.0:9091 com.api.summarizer_api:flask_app
export PYTHONPATH='/Users/mudasir/ShortMail/'
cd ../com/tse
python start.py
