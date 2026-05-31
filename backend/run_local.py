import os
import socket
import sys
from pathlib import Path

import uvicorn


ROOT_DIR = Path(__file__).resolve().parent.parent
FRONTEND_ENV_FILE = ROOT_DIR / "frontend" / ".env.local"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def is_port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return sock.connect_ex((host, port)) != 0


def find_available_port(host: str, preferred_port: int, max_tries: int = 25) -> int:
    for port in range(preferred_port, preferred_port + max_tries):
        if is_port_free(host, port):
            return port
    raise RuntimeError(f"No free port found between {preferred_port} and {preferred_port + max_tries - 1}.")


def write_frontend_env(host: str, port: int) -> None:
    FRONTEND_ENV_FILE.write_text(f"VITE_BACKEND_URL=http://{host}:{port}\n", encoding="utf-8")


def main() -> None:
    host = os.getenv("BACKEND_HOST", "127.0.0.1")
    preferred_port = int(os.getenv("BACKEND_PORT", "8000"))
    port = find_available_port(host, preferred_port)

    if port != preferred_port:
        print(f"Port {preferred_port} is busy. Using port {port} instead.")

    write_frontend_env(host, port)
    print(f"Frontend API target updated in {FRONTEND_ENV_FILE}.")
    print(f"Starting backend at http://{host}:{port}")

    from backend.server import app

    uvicorn.run(app, host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
