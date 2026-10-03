"""Validation of identifiers that end up in tags, unit names, file names and S3 keys."""
import re

RUN_ID = re.compile(r'[A-Za-z0-9-]{1,64}')


def check_run_id(run_id):
    if not isinstance(run_id, str) or not RUN_ID.fullmatch(run_id):
        raise ValueError('run_id must be 1-64 letters, digits or hyphens')
    return run_id
