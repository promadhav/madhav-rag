"""Start/status/stop only this project's worker, with a project-specific socket."""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parent
SOCKET=str(ROOT/'state/worker.sock')


def environment():
    env=os.environ.copy()
    env.update(RAG_SOCKET=SOCKET,RAG_STATE=str(ROOT/'state'),RAG_OUTPUT=str(ROOT/'output'),
               HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    return env


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('action',choices=['start','status','stop'])
    a=ap.parse_args()
    pidfile=ROOT/'state/worker.pid'
    env=environment()
    if a.action=='status':
        raise SystemExit(subprocess.run([sys.executable,str(ROOT/'app.py'),'--health'],env=env).returncode)
    if a.action=='stop':
        if not pidfile.exists():
            print('No project worker PID recorded.')
            return
        pid=int(pidfile.read_text())
        cmd=Path(f'/proc/{pid}/cmdline')
        if cmd.exists() and str(ROOT/'worker.py').encode() in cmd.read_bytes():
            os.kill(pid,signal.SIGTERM)
            print('Stopped project worker',pid)
        else:
            print('Recorded worker is already stopped; no other process touched.')
        pidfile.unlink(missing_ok=True)
        return
    (ROOT/'state').mkdir(exist_ok=True)
    check=subprocess.run([sys.executable,str(ROOT/'app.py'),'--health'],env=env,capture_output=True)
    if check.returncode==0:
        print(check.stdout.decode())
        return
    # An existing live process might still be loading the model.
    if pidfile.exists():
        pid=int(pidfile.read_text())
        cmd=Path(f'/proc/{pid}/cmdline')
        if cmd.exists() and str(ROOT/'worker.py').encode() in cmd.read_bytes():
            print('Worker is still loading. View worker.log, then run status.')
            return
    with (ROOT/'worker.log').open('a') as log:
        proc=subprocess.Popen([sys.executable,'-u',str(ROOT/'worker.py')],cwd=ROOT,env=env,
                               stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    pidfile.write_text(str(proc.pid))
    print('Started worker PID',proc.pid,'. First model load can take a few minutes.')
    print('Logs:',ROOT/'worker.log')
    # Return quickly; the separate --index client waits for readiness.


if __name__=='__main__':
    main()
