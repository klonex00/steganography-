import socket
import numpy as np
import torch
import os
import threading
import struct
import pickle
from cryptography.fernet import Fernet
import ssl
import logging

# === Absolute path for certificates on macOS after renaming folder to 'npsel' ===
BASE_DIR = "xyz"
CERT_DIR = os.path.join(BASE_DIR, 'certificates')

class SecureSteganographyChannel:
    def __init__(self, host: str = '127.0.0.1', port: int = 5000, is_server: bool = False):
        """Initialize a secure channel for steganographic communication."""
        self.host = host
        self.port = port
        self.is_server = is_server

        # Generate a proper Fernet key
        self.key = Fernet.generate_key()
        self.cipher = Fernet(self.key)

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Setup SSL context - create separate contexts for server and client
        if self.is_server:
            self.ssl_context = self._create_server_context()
        else:
            self.ssl_context = self._create_client_context()

    def _create_server_context(self):
        """Create SSL context specifically for server role."""
        try:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.minimum_version = ssl.TLSVersion.TLSv1_2

            # Basic security settings
            context.options |= ssl.OP_NO_SSLv2
            context.options |= ssl.OP_NO_SSLv3
            context.options |= ssl.OP_NO_TLSv1
            context.options |= ssl.OP_NO_TLSv1_1

            # Set basic cipher list
            context.set_ciphers('HIGH:!aNULL:!eNULL:!EXPORT:!SSLv2:!SSLv3:!TLSv1:!TLSv1.1')

            # Server-specific settings
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE

            # Load server certificates using absolute paths
            cert_path = os.path.join(CERT_DIR, 'server.crt')
            key_path = os.path.join(CERT_DIR, 'server.key')

            if not os.path.exists(cert_path) or not os.path.exists(key_path):
                raise FileNotFoundError(f"Server certificates not found at {cert_path} or {key_path}")

            context.load_cert_chain(certfile=cert_path, keyfile=key_path)
            return context

        except Exception as e:
            self.logger.error(f"Error creating server SSL context: {e}")
            raise

    def _create_client_context(self):
        """Create SSL context specifically for client role."""
        try:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            context.minimum_version = ssl.TLSVersion.TLSv1_2

            # Basic security settings
            context.options |= ssl.OP_NO_SSLv2
            context.options |= ssl.OP_NO_SSLv3
            context.options |= ssl.OP_NO_TLSv1
            context.options |= ssl.OP_NO_TLSv1_1

            # Set basic cipher list
            context.set_ciphers('HIGH:!aNULL:!eNULL:!EXPORT:!SSLv2:!SSLv3:!TLSv1:!TLSv1.1')

            # Client-specific settings
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE

            # If client needs its own cert/key for mutual auth, uncomment below:
            # client_cert_path = os.path.join(CERT_DIR, 'client.crt')
            # client_key_path = os.path.join(CERT_DIR, 'client.key')
            # if os.path.exists(client_cert_path) and os.path.exists(client_key_path):
            #     context.load_cert_chain(certfile=client_cert_path, keyfile=client_key_path)

            return context

        except Exception as e:
            self.logger.error(f"Error creating client SSL context: {e}")
            raise

    def send_tensor(self, sock: ssl.SSLSocket, tensor: torch.Tensor):
        """Send a tensor securely over the SSL socket."""
        try:
            tensor = tensor.detach().cpu()
            if len(tensor.shape) == 4:
                tensor = tensor.squeeze(0)
            data = pickle.dumps(tensor.numpy())
            encrypted_data = self.cipher.encrypt(data)

            # Send key first (only for the first tensor)
            if not hasattr(self, '_key_sent'):
                key_length = len(self.key)
                sock.sendall(struct.pack('!I', key_length))
                sock.sendall(self.key)
                self._key_sent = True

            # Send encrypted data length
            length = len(encrypted_data)
            sock.sendall(struct.pack('!Q', length))

            # Send encrypted data in chunks
            chunk_size = 8192
            for i in range(0, length, chunk_size):
                chunk = encrypted_data[i:i + chunk_size]
                sock.sendall(chunk)

        except Exception as e:
            self.logger.error(f"Error sending tensor: {e}")
            raise

    def receive_tensor(self, sock: ssl.SSLSocket) -> torch.Tensor:
        """Receive a tensor securely from the SSL socket."""
        try:
            # Receive key first (only for the first tensor)
            if not hasattr(self, '_key_received'):
                key_length = struct.unpack('!I', sock.recv(4))[0]
                self.key = sock.recv(key_length)
                self.cipher = Fernet(self.key)
                self._key_received = True

            # Receive encrypted data length
            length_bytes = sock.recv(8)
            if not length_bytes:
                raise ConnectionError("Connection closed")
            length = struct.unpack('!Q', length_bytes)[0]

            # Receive encrypted data in chunks
            encrypted_data = bytearray()
            while len(encrypted_data) < length:
                chunk = sock.recv(min(8192, length - len(encrypted_data)))
                if not chunk:
                    raise ConnectionError("Connection closed")
                encrypted_data.extend(chunk)

            # Decrypt data
            data = self.cipher.decrypt(bytes(encrypted_data))

            # Convert to tensor
            array = pickle.loads(data)
            tensor = torch.from_numpy(array)
            if len(tensor.shape) == 3:
                tensor = tensor.unsqueeze(0)
            return tensor

        except Exception as e:
            self.logger.error(f"Error receiving tensor: {e}")
            raise

    def start_server(self):
        """Start the secure steganography server with SSL/TLS."""
        server_socket = None
        ssl_server_socket = None
        try:
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((self.host, self.port))
            server_socket.listen(1)

            if not isinstance(self.ssl_context, ssl.SSLContext) or self.ssl_context.protocol != ssl.PROTOCOL_TLS_SERVER:
                raise ValueError("Invalid SSL context for server role")

            ssl_server_socket = self.ssl_context.wrap_socket(server_socket, server_side=True)
            self.logger.info(f"Secure server listening on {self.host}:{self.port}")

            while True:
                try:
                    client_socket, addr = ssl_server_socket.accept()
                    self.logger.info(f"Accepted connection from {addr}")
                    if hasattr(self, '_key_sent'):
                        delattr(self, '_key_sent')
                    client_thread = threading.Thread(target=self._handle_client, args=(client_socket,))
                    client_thread.start()
                except ssl.SSLError as e:
                    self.logger.error(f"SSL Error accepting connection: {e}")
                    continue
                except Exception as e:
                    self.logger.error(f"Error accepting connection: {e}")
                    continue

        except Exception as e:
            self.logger.error(f"Server error: {e}")
            raise
        finally:
            if ssl_server_socket:
                try: ssl_server_socket.close()
                except: pass
            if server_socket:
                try: server_socket.close()
                except: pass

    def connect_to_server(self) -> ssl.SSLSocket:
        """Connect to the secure steganography server using SSL/TLS."""
        sock = None
        ssl_sock = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((self.host, self.port))
            if not isinstance(self.ssl_context, ssl.SSLContext) or self.ssl_context.protocol != ssl.PROTOCOL_TLS_CLIENT:
                raise ValueError("Invalid SSL context for client role")
            ssl_sock = self.ssl_context.wrap_socket(sock, server_side=False)
            self.logger.info("Connected to secure server")
            if hasattr(self, '_key_received'):
                delattr(self, '_key_received')
            return ssl_sock
        except ssl.SSLError as e:
            self.logger.error(f"SSL Error connecting to server: {e}")
            if ssl_sock:
                try: ssl_sock.close()
                except: pass
            if sock:
                try: sock.close()
                except: pass
            raise
        except Exception as e:
            self.logger.error(f"Error connecting to server: {e}")
            if ssl_sock:
                try: ssl_sock.close()
                except: pass
            if sock:
                try: sock.close()
                except: pass
            raise

    def _handle_client(self, client_socket: ssl.SSLSocket):
        """Handle client communication in server mode."""
        try:
            # Overridden by demo’s handle_client
            pass
        finally:
            client_socket.close()

# Example usage (testing):
if __name__ == "__main__":
    server = SecureSteganographyChannel(is_server=True)
    threading.Thread(target=server.start_server, daemon=True).start()
    time.sleep(1)
    client = SecureSteganographyChannel(is_server=False)
    sock = client.connect_to_server()
    test_tensor = torch.randn(1, 3, 128, 128)
    client.send_tensor(sock, test_tensor)
    received = client.receive_tensor(sock)
    print("Secure test completed.")
    sock.close()
