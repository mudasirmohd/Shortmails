#!/usr/bin/env bash
export PYTHONPATH=/home/shabir/p-work/personal_work/quickwordz
export PATH=com/bin/:$PATH
gunicorn -w 3 --timeout 200 -b 0.0.0.0:3000 com.api.infer_api:flask_app