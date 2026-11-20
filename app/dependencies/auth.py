"""Shared service credential; no user/session authentication."""
import secrets

from fastapi import Depends, HTTPException, Request
from fastapi.security import APIKeyHeader
from starlette.responses import JSONResponse

scheme = APIKeyHeader(name='X-App-Secret-Key', scheme_name='AppSecret', auto_error=False)


def valid_secret(candidate: str | None, expected: str) -> bool:
    return candidate is not None and secrets.compare_digest(candidate.encode('utf-8'), expected.encode('utf-8'))


def require_app_secret(request: Request, key: str | None = Depends(scheme)) -> None:
    if not valid_secret(key, request.app.state.config.require_api().secret_key):
        raise HTTPException(401, 'Invalid service credential')


class ServiceBoundary:
    """Reject unauthenticated bodies before multipart parsing or dependency execution."""
    def __init__(self, app, *, secret: str, max_upload_bytes: int):
        self.app, self.secret, self.max_bytes = app, secret, max_upload_bytes + 1024 * 1024

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or not scope['path'].startswith('/api/'):
            return await self.app(scope, receive, send)
        headers = scope.get('headers', [])
        keys = [value.decode('latin-1') for key, value in headers if key.lower() == b'x-app-secret-key']
        if len(keys) != 1 or not valid_secret(keys[0], self.secret):
            return await JSONResponse({'detail': 'Invalid service credential'}, status_code=401)(scope, receive, send)
        received = 0
        async def limited_receive():
            nonlocal received
            message = await receive()
            received += len(message.get('body', b''))
            if received > self.max_bytes:
                raise HTTPException(413, 'Request exceeds upload limit')
            return message
        try:
            await self.app(scope, limited_receive, send)
        except Exception as error:
            # Do not allow Uvicorn to log raw exception tracebacks containing driver/provider data.
            from app.api import error_response
            response = await error_response(Request(scope), error)
            await response(scope, receive, send)
