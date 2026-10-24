"""Harness-compatible lightweight client. No torch imports or reindex on queries."""
import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

SOCKET = os.environ.get('RAG_SOCKET', '/tmp/madhav-rag.sock')
EMPTY = {'answer': '', 'citations': []}


def rpc(payload, timeout):
    deadline = time.monotonic() + timeout
    while True:
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(max(0.01, deadline-time.monotonic()))
            sock.connect(SOCKET)
            break
        except (FileNotFoundError, ConnectionRefusedError):
            sock.close()
            if time.monotonic() >= deadline:
                raise TimeoutError('Worker not ready. Check worker.log.')
            time.sleep(0.15)
    with sock:
        sock.sendall(json.dumps(payload).encode() + b'\n')
        data = bytearray()
        while b'\n' not in data:
            sock.settimeout(max(0.01, deadline-time.monotonic()))
            part = sock.recv(65536)
            if not part:
                raise ConnectionError('Worker disconnected')
            data.extend(part)
            if len(data) > 4_000_000:
                raise ValueError('Unexpected response size')
        result = json.loads(bytes(data).split(b'\n',1)[0])
    if 'error' in result:
        raise RuntimeError(result['error'])
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--index')
    p.add_argument('--corpus')
    p.add_argument('--query-id')
    p.add_argument('--query')
    p.add_argument('--health', action='store_true')
    a = p.parse_args()
    if a.health:
        print(json.dumps(rpc({'op':'health'}, 2), indent=2))
        return
    if a.index:
        if a.corpus or a.query is not None:
            p.error('--index is a separate invocation')
        print(json.dumps(rpc({'op':'index','corpus':str(Path(a.index).resolve())}, 570),indent=2))
        return
    if not a.corpus or a.query is None or not a.query_id:
        p.error('--corpus, --query-id and --query are required')
    if a.query_id in ('.','..') or any(c in a.query_id for c in '/\\\x00'):
        p.error('query-id must be a filename stem')
    out = Path(os.environ.get('RAG_OUTPUT', '/app/output'))
    out.mkdir(parents=True, exist_ok=True)
    target = out / (a.query_id + '_output.json')
    started = time.monotonic()
    result, failed = EMPTY.copy(), False
    try:
        result = rpc({'op':'query','corpus':str(Path(a.corpus).resolve()),
                      'query':a.query}, 27)
        if not isinstance(result.get('answer'), str) or not isinstance(result.get('citations'),list):
            raise ValueError('Malformed worker response')
    except Exception as e:
        # Always write the expected JSON; log failures instead of calling them successful refusals.
        result, failed = EMPTY.copy(), True
        print(f'QUERY ERROR: {e}', file=sys.stderr)
    tmp = target.with_name(target.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(result,ensure_ascii=False),encoding='utf-8')
    tmp.replace(target)
    print(json.dumps({'file':str(target),'seconds':round(time.monotonic()-started,3),**result}))
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
