"""Loopback provider server shared by HTTP and database pipeline integrations."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from threading import Thread
import time

import pytest


@pytest.fixture
def server():
    state = {'requests': [], 'status': 200, 'delay': 0}
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, body):
            time.sleep(state['delay'])
            self.send_response(state['status'])
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            if state['status'] != 200:
                body = {'error': {'message': 'secret upstream diagnostic', 'type': 'test', 'code': 'test'}}
            try:
                self.wfile.write(json.dumps(body).encode())
            except BrokenPipeError:
                pass

        def do_GET(self):
            state['requests'].append((self.path, None, self.headers.get('Authorization')))
            self.respond({'data': [{'id': 'remote-model', 'object': 'model', 'created': 0, 'owned_by': 'test'}]})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            state['requests'].append((self.path, body, self.headers.get('Authorization')))
            if self.path.endswith('/embeddings'):
                response = {'object': 'list', 'model': body['model'], 'data': [
                    {'object': 'embedding', 'index': i, 'embedding': [1., 0.]}
                    for i in reversed(range(len(body['input'])))],
                    'usage': {'prompt_tokens': 1, 'total_tokens': 1}}
            else:
                response = {'id': 'chat-id', 'object': 'chat.completion', 'created': 0, 'model': body['model'],
                            'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'answer'},
                                         'finish_reason': 'stop'}],
                            'usage': {'prompt_tokens': 2, 'completion_tokens': 1, 'total_tokens': 3}}
            self.respond(response)
    http = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = Thread(target=http.serve_forever, daemon=True)
    thread.start()
    state['url'] = f'http://127.0.0.1:{http.server_port}'
    try:
        yield state
    finally:
        http.shutdown()
        http.server_close()
        thread.join()


