import sys
from fakeredis import TcpFakeServer

def main():
    server_address = ("127.0.0.1", 6379)
    print(f"Starting mock Redis server on {server_address[0]}:{server_address[1]}...", flush=True)
    try:
        server = TcpFakeServer(server_address, server_type="redis")
        print("Mock Redis server is running. Press Ctrl+C to stop.", flush=True)
        server.serve_forever()
    except Exception as e:
        print(f"Error starting mock Redis: {e}", file=sys.stderr, flush=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
