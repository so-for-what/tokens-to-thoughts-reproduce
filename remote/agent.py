#!/usr/bin/env python3
"""
Hermes Relay Agent — remote command execution for USTC 107 platform.

Usage (inside Apptainer on compute node):
  python3 remote/agent.py

Connects to the relay server via Tailscale Funnel, polls for commands,
executes them, and posts results back.
"""

import urllib.request as u
import json
import subprocess as s
import time
import sys
import traceback

RELAY = 'https://laptop-mn6kbpmm.tail39c976.ts.net:8443/hermes'
LOG = '/tmp/agent_debug.log'


def log(msg):
    with open(LOG, 'a') as f:
        f.write(f'[{time.strftime("%H:%M:%S")}] {msg}\n')
    print(msg, flush=True)


def get(endpoint):
    try:
        resp = u.urlopen(RELAY + endpoint, timeout=10)
        data = json.loads(resp.read().decode())
        return data
    except Exception as ex:
        log(f'GET {endpoint} FAILED: {ex}')
        return {}


def post(endpoint, data_dict):
    try:
        body = json.dumps(data_dict).encode()
        req = u.Request(
            RELAY + endpoint,
            data=body,
            headers={'Content-Type': 'application/json'},
        )
        resp = u.urlopen(req, timeout=10)
        log(f'POST {endpoint} -> {resp.status}')
        return True
    except Exception as ex:
        log(f'POST {endpoint} FAILED: {ex}')
        return False


def main():
    post('/register', {})
    log('Agent started')

    while True:
        try:
            c = get('/next')
            if c and c.get('cmd'):
                cmd = c['cmd']
                log(f'Executing: {cmd[:80]}')
                try:
                    r = s.run(
                        cmd, shell=True,
                        capture_output=True, text=True, timeout=600,
                    )
                    result = {
                        'stdout': r.stdout[-4000:],
                        'stderr': r.stderr[-4000:],
                        'exit': r.returncode,
                    }
                    log(f'Exit: {r.returncode}')
                    ok = post('/result/' + c['id'], result)
                    log(f'Result posted: {ok}')
                except s.TimeoutExpired:
                    log('TIMEOUT')
                    post('/result/' + c['id'], {
                        'error': 'timeout',
                    })
                except Exception as ex:
                    log(f'EXEC ERROR: {ex}')
                    post('/result/' + c['id'], {
                        'error': str(ex),
                    })
            else:
                time.sleep(1)
        except Exception as ex:
            log(f'LOOP ERROR: {ex}\n{traceback.format_exc()}')
            time.sleep(5)


if __name__ == '__main__':
    main()