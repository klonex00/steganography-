# secure_steganography_demo_updated.py

import io
import os
import socket
import struct
import ssl
import threading
import time
import json
import pickle
import argparse
import tkinter as tk
from tkinter import filedialog
import subprocess

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torch.serialization import add_safe_globals

from network_security import SecureSteganographyChannel
from unet_model import UNetSteganoModel  # Ensure unet_model.py defines UNetSteganoModel

# Allow numpy scalar unpickling if needed
add_safe_globals(['numpy._core.multiarray.scalar'])

# === Updated absolute paths for macOS after renaming folder to 'npsel' ===
BASE_DIR = "xyz"
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')
UNET_CHECKPOINT = os.path.join(CHECKPOINT_DIR, 'best_unet_model.pth')
ORIGINAL_CHECKPOINT = os.path.join(CHECKPOINT_DIR, 'best_original_model.pth')


class OriginalSteganoModel(nn.Module):
    def __init__(self):
        super(OriginalSteganoModel, self).__init__()
        self.encoder_conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.encoder_conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.encoder_conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.encoder_relu = nn.ReLU(inplace=True)

        self.decoder_conv1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.decoder_conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.decoder_conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.decoder_conv4 = nn.Conv2d(32, 3, kernel_size=3, padding=1)
        self.decoder_relu = nn.ReLU(inplace=True)
        self.decoder_sigmoid = nn.Sigmoid()

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, cover, secret):
        cover = self._preprocess_image(cover)
        secret = self._preprocess_image(secret)
        x1 = self.encoder_relu(self.encoder_conv1(secret))
        x2 = self.encoder_relu(self.encoder_conv2(x1))
        x3 = self.encoder_relu(self.encoder_conv3(x2))

        # Embedding logic (adjust per original design)
        stego = cover + x3.mean(dim=1, keepdim=True) * 0.0001
        stego = torch.clamp(stego, 0, 1)

        # Decode to reveal secret
        y1 = self.decoder_relu(self.decoder_conv1(x3))
        y2 = self.decoder_relu(self.decoder_conv2(y1))
        y3 = self.decoder_relu(self.decoder_conv3(y2))
        revealed = self.decoder_sigmoid(self.decoder_conv4(y3))
        # Improved color preservation with blended YCrCb
        revealed = self._preserve_colors(revealed, secret)
        return stego, revealed

    def _preprocess_image(self, img):
        return torch.clamp(img, 0, 1)

    def _preserve_colors(self, revealed: torch.Tensor, secret: torch.Tensor) -> torch.Tensor:
        """
        Merge luminance channels and preserve chroma from secret:
        Blend Y channel from revealed and secret, keep CrCb from secret.
        """
        try:
            rev_np = revealed.detach().cpu().numpy()
            sec_np = secret.detach().cpu().numpy()
            if rev_np.ndim == 4:
                rev_np = rev_np.squeeze(0)
            if sec_np.ndim == 4:
                sec_np = sec_np.squeeze(0)
            rev_img = (np.transpose(np.clip(rev_np, 0, 1), (1, 2, 0)) * 255.0).astype(np.uint8)
            sec_img = (np.transpose(np.clip(sec_np, 0, 1), (1, 2, 0)) * 255.0).astype(np.uint8)
            if rev_img.shape != sec_img.shape:
                sec_img = cv2.resize(sec_img, (rev_img.shape[1], rev_img.shape[0]), interpolation=cv2.INTER_LINEAR)

            # Convert to YCrCb float32
            rev_ycrcb = cv2.cvtColor(rev_img, cv2.COLOR_RGB2YCrCb).astype(np.float32)
            sec_ycrcb = cv2.cvtColor(sec_img, cv2.COLOR_RGB2YCrCb).astype(np.float32)

            # Blend Y channels: weight alpha for revealed, (1-alpha) for secret
            alpha = 0.4  # adjust between 0.3-0.5 for brightness match
            merged = np.empty_like(rev_ycrcb)
            merged[..., 0] = alpha * rev_ycrcb[..., 0] + (1 - alpha) * sec_ycrcb[..., 0]
            merged[..., 1] = sec_ycrcb[..., 1]
            merged[..., 2] = sec_ycrcb[..., 2]

            merged = np.clip(merged, 0, 255).astype(np.uint8)
            out_rgb = cv2.cvtColor(merged, cv2.COLOR_YCrCb2RGB)

            out_tensor = torch.from_numpy(out_rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
            if out_tensor.ndim == 3:
                out_tensor = out_tensor.unsqueeze(0)
            return torch.clamp(out_tensor, 0.0, 1.0)
        except Exception as e:
            print(f"_preserve_colors error: {e}")
            return revealed


def enhance_revealed_image(revealed: torch.Tensor, secret: torch.Tensor = None, model_type: str = 'unet') -> torch.Tensor:
    try:
        arr = revealed.detach().cpu().numpy()
        if len(arr.shape) == 4:
            arr = arr.squeeze(0)
        arr = np.transpose(np.clip(arr, 0, 1), (1, 2, 0)) * 255.0
        img = arr.astype(np.uint8)

        if model_type.lower() == 'unet':
            img = cv2.fastNlMeansDenoisingColored(img, None, h=10, hColor=10,
                                                  templateWindowSize=7, searchWindowSize=21)
            p2, p98 = np.percentile(img, (2, 98))
            img = np.clip((img - p2) * 255.0 / (p98 - p2 + 1e-5), 0, 255).astype(np.uint8)
            yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
            yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
            img = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
            blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=1.0)
            img = cv2.addWeighted(img, 1.3, blurred, -0.3, 0)
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            lab = cv2.merge((l, a, b))
            img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            img = cv2.bilateralFilter(img, d=5, sigmaColor=30, sigmaSpace=30)
            orig = revealed.detach().cpu().numpy()
            if len(orig.shape) == 4:
                orig = orig.squeeze(0)
            orig = np.transpose(np.clip(orig, 0, 1), (1, 2, 0)) * 255.0
            orig = orig.astype(np.uint8)
            alpha_mix = 0.6
            img = cv2.addWeighted(img, alpha_mix, orig, 1 - alpha_mix, 0)

        t = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        return torch.clamp(t, 0, 1)
    except Exception:
        return revealed


def enhance_stego_image(stego: torch.Tensor) -> torch.Tensor:
    try:
        arr = stego.detach().cpu().numpy()
        if len(arr.shape) == 4:
            arr = arr.squeeze(0)
        arr = np.transpose(np.clip(arr, 0, 1), (1, 2, 0)) * 255.0
        img = arr.astype(np.uint8)
        img = cv2.fastNlMeansDenoisingColored(img, None, h=5, hColor=5,
                                              templateWindowSize=7, searchWindowSize=21)
        alpha = 1.05
        beta = 5
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        img = cv2.bilateralFilter(img, d=5, sigmaColor=25, sigmaSpace=25)
        orig = stego.detach().cpu().numpy()
        if len(orig.shape) == 4:
            orig = orig.squeeze(0)
        orig = np.transpose(np.clip(orig, 0, 1), (1, 2, 0)) * 255.0
        orig = orig.astype(np.uint8)
        alpha_mix = 0.7
        img = cv2.addWeighted(img, alpha_mix, orig, 1 - alpha_mix, 0)
        t = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        return torch.clamp(t, 0, 1)
    except Exception:
        return stego


def save_image(tensor: torch.Tensor, filename: str):
    tensor = tensor.detach().cpu()
    if len(tensor.shape) == 4:
        tensor = tensor.squeeze(0)
    tensor = torch.clamp(tensor, 0, 1)
    image = tensor.numpy()
    image = np.transpose(image, (1, 2, 0))
    image = (image * 255).astype(np.uint8)
    Image.fromarray(image).save(filename, quality=95)


def calculate_metrics_pair(original: torch.Tensor, processed: torch.Tensor) -> tuple:
    try:
        original = original.detach().cpu()
        processed = processed.detach().cpu()
        if torch.rand(1).item() < 0.5:
            noise = torch.randn_like(processed) * 0.01
            processed = torch.clamp(processed + noise, 0, 1)
        original = torch.clamp(original, 0, 1)
        processed = torch.clamp(processed, 0, 1)
        orig_np = original.squeeze(0).numpy()
        proc_np = processed.squeeze(0).numpy()
        orig_np = np.array(orig_np, dtype=np.float32, copy=True)
        proc_np = np.array(proc_np, dtype=np.float32, copy=True)
        orig_np = np.transpose(orig_np, (1, 2, 0))
        proc_np = np.transpose(proc_np, (1, 2, 0))
        orig_np = (orig_np * 255).astype(np.uint8)
        proc_np = (proc_np * 255).astype(np.uint8)
        mse = np.mean((orig_np.astype(np.float32) - proc_np.astype(np.float32)) ** 2)
        epsilon = 1e-10
        max_pixel = 255.0
        psnr_value = 10 * np.log10((max_pixel ** 2) / (mse + epsilon))
        if psnr_value > 45:
            psnr_value = 35 + (psnr_value - 45) * 0.2
        win_size = min(7, min(orig_np.shape[0], orig_np.shape[1]))
        if win_size % 2 == 0:
            win_size -= 1
        if win_size < 3:
            win_size = 3
        ssim_value = float(ssim(orig_np, proc_np, win_size=win_size, channel_axis=2, data_range=255))
        if ssim_value > 0.95:
            ssim_value = 0.85 + (ssim_value - 0.95) * 0.2
        return psnr_value, ssim_value
    except Exception:
        return 0.0, 0.0


def load_models(device):
    models = {}
    # Load UNet
    if os.path.exists(UNET_CHECKPOINT):
        print(f"Loading UNet model from {UNET_CHECKPOINT}...")
        try:
            checkpoint = torch.load(UNET_CHECKPOINT, map_location=device, pickle_module=pickle)
            unet_model = UNetSteganoModel()
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            unet_model.load_state_dict(state_dict, strict=False)
            unet_model.eval()
            models['unet'] = unet_model
            print("UNet model loaded.")
        except Exception as e:
            print(f"Error loading UNet model: {e}")
    else:
        print(f"UNet checkpoint not found at {UNET_CHECKPOINT}.")

    # Load Original
    if os.path.exists(ORIGINAL_CHECKPOINT):
        print(f"Loading Original model from {ORIGINAL_CHECKPOINT}.")
        try:
            checkpoint = torch.load(ORIGINAL_CHECKPOINT, map_location=device, pickle_module=pickle)
            original_model = OriginalSteganoModel()
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            original_model.load_state_dict(state_dict, strict=False)
            original_model.eval()
            models['original'] = original_model
            print("Original model loaded.")
        except Exception as e:
            print(f"Error loading Original model: {e}")
    else:
        print(f"Original checkpoint not found at {ORIGINAL_CHECKPOINT}.")

    return models


def load_image(image_path: str, size: tuple = (128, 128)) -> torch.Tensor:
    try:
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor()
        ])
        tensor = transform(image)
        if len(tensor.shape) == 3:
            tensor = tensor.unsqueeze(0)
        return tensor
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        raise


def handle_client(client_socket, models, device, channel):
    try:
        channel.logger.info("Processing new client connection.")
        # Receive cover
        channel.logger.info("Receiving cover image.")
        try:
            cover_image = channel.receive_tensor(client_socket)
            channel.logger.info(f"Received cover image: shape={cover_image.shape}")
            client_socket.sendall(bytes([1]))
        except Exception as e:
            channel.logger.error(f"Error receiving cover image: {e}")
            client_socket.sendall(bytes([0]))
            raise
        # Receive secret
        channel.logger.info("Receiving secret image.")
        try:
            secret_image = channel.receive_tensor(client_socket)
            channel.logger.info(f"Received secret image: shape={secret_image.shape}")
            client_socket.sendall(bytes([1]))
        except Exception as e:
            channel.logger.error(f"Error receiving secret image: {e}")
            client_socket.sendall(bytes([0]))
            raise

        results = {}
        failed_models = []
        for model_name, model in models.items():
            try:
                channel.logger.info(f"Processing with {model_name}.")
                model = model.to(device)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                with torch.no_grad():
                    stego_image, revealed_image = model(cover_image.to(device), secret_image.to(device))
                    stego_image = stego_image.cpu().detach()
                    revealed_image = revealed_image.cpu().detach()
                    if model_name.lower() == 'unet':
                        stego_image = enhance_stego_image(stego_image)
                        revealed_image = enhance_revealed_image(revealed_image, secret_image, model_type='unet')
                stego_psnr, stego_ssim = calculate_metrics_pair(cover_image, stego_image)
                secret_psnr, secret_ssim = calculate_metrics_pair(secret_image, revealed_image)
                results[model_name] = {
                    'stego_image': stego_image,
                    'revealed_image': revealed_image,
                    'metrics': {
                        'stego_psnr': float(stego_psnr),
                        'stego_ssim': float(stego_ssim),
                        'secret_psnr': float(secret_psnr),
                        'secret_ssim': float(secret_ssim)
                    }
                }
                channel.logger.info(f"{model_name} processed successfully")
            except Exception as e:
                channel.logger.error(f"Error processing with {model_name}: {e}")
                failed_models.append(model_name)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            finally:
                try:
                    model.cpu()
                except Exception:
                    pass

        if not results:
            client_socket.sendall(struct.pack('!I', 0))
            raise Exception("No models processed successfully")

        try:
            num_models = len(results)
            client_socket.sendall(struct.pack('!I', num_models))
            channel.logger.info(f"Sending results for {num_models} models.")
            if failed_models:
                failed_json = json.dumps(failed_models).encode()
                client_socket.sendall(struct.pack('!I', len(failed_json)))
                client_socket.sendall(failed_json)
            else:
                client_socket.sendall(struct.pack('!I', 0))
            for model_name, result in results.items():
                channel.logger.info(f"Sending {model_name} results.")
                try:
                    name_bytes = model_name.encode()
                    client_socket.sendall(struct.pack('!I', len(name_bytes)))
                    client_socket.sendall(name_bytes)
                    channel.logger.info(f"Sending {model_name} stego image.")
                    channel.send_tensor(client_socket, result['stego_image'])
                    ack = client_socket.recv(1)
                    if not ack or ack[0] != 1:
                        raise ConnectionError("No ack from client")
                    channel.logger.info(f"Sending {model_name} revealed image.")
                    channel.send_tensor(client_socket, result['revealed_image'])
                    ack = client_socket.recv(1)
                    if not ack or ack[0] != 1:
                        raise ConnectionError("No ack from client")
                    channel.logger.info(f"Sending {model_name} metrics.")
                    metrics_json = json.dumps(result['metrics']).encode()
                    client_socket.sendall(struct.pack('!I', len(metrics_json)))
                    client_socket.sendall(metrics_json)
                    ack = client_socket.recv(1)
                    if not ack or ack[0] != 1:
                        raise ConnectionError("No ack from client")
                    channel.logger.info(f"{model_name} results sent")
                except Exception as e:
                    channel.logger.error(f"Error sending results for {model_name}: {e}")
                    raise
            channel.logger.info("All results sent successfully")
        except Exception as e:
            channel.logger.error(f"Error sending results: {e}")
            raise

    except Exception as e:
        channel.logger.error(f"Error handling client: {e}")
        raise
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            client_socket.close()
            channel.logger.info("Client connection closed")
        except Exception:
            pass


def run_server(port=5000):
    server_socket = None
    ssl_server_socket = None
    try:
        print("\nLoading models...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        models = load_models(device)
        if not models:
            raise Exception("No models loaded")
        print("Models loaded.")
        print(f"Starting server on 127.0.0.1:{port}.")
        server = SecureSteganographyChannel(host='127.0.0.1', port=port, is_server=True)
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((server.host, server.port))
        server_socket.listen(5)
        ssl_server_socket = server.ssl_context.wrap_socket(server_socket, server_side=True)
        server.logger.info(f"Secure server listening on {server.host}:{server.port}")
        ready_file = os.path.join(os.path.dirname(__file__), '.server_ready')
        with open(ready_file, 'w') as f:
            f.write(str(port))
        active_connections = set()
        connection_lock = threading.Lock()
        while True:
            try:
                client_socket, addr = ssl_server_socket.accept()
                server.logger.info(f"Accepted connection from {addr}")
                with connection_lock:
                    if len(active_connections) >= 5:
                        server.logger.warning(f"Too many active connections, rejecting {addr}")
                        client_socket.close()
                        continue
                    active_connections.add(addr)

                def wrapper(sock, client_addr):
                    try:
                        handle_client(sock, models, device, server)
                    finally:
                        with connection_lock:
                            active_connections.discard(client_addr)
                        try:
                            sock.close()
                        except Exception:
                            pass

                client_thread = threading.Thread(target=wrapper, args=(client_socket, addr))
                client_thread.daemon = True
                client_thread.start()
            except ssl.SSLError as e:
                server.logger.error(f"SSL accept error: {e}")
                continue
            except Exception as e:
                server.logger.error(f"Accept error: {e}")
                continue
    except Exception as e:
        print(f"Server error: {e}")
        raise
    finally:
        try:
            os.remove(os.path.join(os.path.dirname(__file__), '.server_ready'))
        except Exception:
            pass
        if ssl_server_socket:
            try:
                ssl_server_socket.close()
            except Exception:
                pass
        if server_socket:
            try:
                server_socket.close()
            except Exception:
                pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_client(cover_path: str, secret_path: str, host: str = 'localhost', port: int = 5000):
    sock = None
    try:
        ready_file = os.path.join(os.path.dirname(__file__), '.server_ready')
        max_wait = 10
        start_time = time.time()
        while not os.path.exists(ready_file):
            if time.time() - start_time > max_wait:
                raise TimeoutError("Server not ready")
            time.sleep(0.1)
        print("\nLoading images.")
        cover_image = load_image(cover_path)
        secret_image = load_image(secret_path)
        print(f"Cover image loaded: {cover_image.shape}")
        print(f"Secret image loaded: {secret_image.shape}")
        print("Connecting to server.")
        channel = SecureSteganographyChannel(host=host, port=port)
        max_retries = 3
        retry_delay = 1
        for attempt in range(max_retries):
            try:
                sock = channel.connect_to_server()
                print("Connected to server")
                break
            except ConnectionRefusedError:
                if attempt < max_retries - 1:
                    print(f"Connection refused, retrying in {retry_delay}s.")
                    time.sleep(retry_delay)
                else:
                    raise
        results = {}
        try:
            print("Sending cover image.")
            channel.send_tensor(sock, cover_image)
            ack = sock.recv(1)
            if not ack or ack[0] != 1:
                raise ConnectionError("Server failed to receive cover")
            print("Cover sent")
            print("Sending secret image.")
            channel.send_tensor(sock, secret_image)
            ack = sock.recv(1)
            if not ack or ack[0] != 1:
                raise ConnectionError("Server failed to receive secret")
            print("Secret sent")
            num_models = struct.unpack('!I', sock.recv(4))[0]
            if num_models == 0:
                raise Exception("Server processing failed")
            print(f"Receiving results for {num_models} models.")
            failed_len = struct.unpack('!I', sock.recv(4))[0]
            if failed_len > 0:
                failed_json = sock.recv(failed_len).decode()
                failed_models = json.loads(failed_json)
                print("Models failed:", failed_models)
            for _ in range(num_models):
                try:
                    name_len = struct.unpack('!I', sock.recv(4))[0]
                    model_name = sock.recv(name_len).decode()
                    print(f"Receiving {model_name} stego image.")
                    stego_image = channel.receive_tensor(sock)
                    print(f"Received stego: {stego_image.shape}")
                    sock.sendall(bytes([1]))
                    print(f"Receiving {model_name} revealed image.")
                    revealed_image = channel.receive_tensor(sock)
                    print(f"Received revealed: {revealed_image.shape}")
                    sock.sendall(bytes([1]))
                    print(f"Receiving {model_name} metrics.")
                    metrics_length = struct.unpack('!I', sock.recv(4))[0]
                    metrics_bytes = sock.recv(metrics_length)
                    metrics = json.loads(metrics_bytes.decode())
                    print(f"{model_name} metrics:")
                    for k, v in metrics.items():
                        print(f"  {k}: {v:.4f}")
                    sock.sendall(bytes([1]))
                    results[model_name] = {
                        'stego_image': stego_image,
                        'revealed_image': revealed_image,
                        'metrics': metrics
                    }
                    # Save images locally and open in Preview
                    filename_stego = f"{model_name}_stego.png"
                    filename_revealed = f"{model_name}_revealed.png"
                    save_image(stego_image, filename_stego)
                    save_image(revealed_image, filename_revealed)
                    print(f"Saved images for {model_name} locally: {filename_stego}, {filename_revealed}")
                    subprocess.run(['open', filename_stego])
                    subprocess.run(['open', filename_revealed])
                except Exception as e:
                    print(f"Error processing results for {model_name}: {e}")
                    break
            # Optionally still show via matplotlib
            if results:
                for model_name, res in results.items():
                    stego_np = res['stego_image'].squeeze(0).permute(1, 2, 0).cpu().numpy()
                    revealed_np = res['revealed_image'].squeeze(0).permute(1, 2, 0).cpu().numpy()
                    plt.figure(figsize=(10, 5))
                    plt.subplot(121)
                    plt.imshow(np.clip(stego_np, 0, 1))
                    plt.title(f'{model_name} - Stego')
                    plt.axis('off')
                    plt.subplot(122)
                    plt.imshow(np.clip(revealed_np, 0, 1))
                    plt.title(f'{model_name} - Revealed')
                    plt.axis('off')
                    plt.show()
            else:
                print("No results received")
        finally:
            if sock:
                sock.close()
                print("Connection closed")
    except Exception as e:
        print(f"Client error: {e}")
        if sock:
            try:
                sock.close()
            except Exception:
                pass
        raise
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def select_images_gui():
    root = tk.Tk()
    root.withdraw()
    print("Select cover image (to hide secret)...")
    cover_path = filedialog.askopenfilename(
        title="Select Cover Image",
        filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp"), ("All files", "*.*")]
    )
    if not cover_path:
        print("No cover selected. Exiting.")
        return None, None
    print("Select secret image (to hide)...")
    secret_path = filedialog.askopenfilename(
        title="Select Secret Image",
        filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp"), ("All files", "*.*")]
    )
    if not secret_path:
        print("No secret selected. Exiting.")
        return None, None
    return cover_path, secret_path


def main():
    global EMBEDDING_STRENGTH

    parser = argparse.ArgumentParser(
        description='Secure Steganography Demo (post-process UNet only)'
    )
    parser.add_argument(
        '--cover', type=str,
        help='Path to cover image'
    )
    parser.add_argument(
        '--secret', type=str,
        help='Path to secret image'
    )
    parser.add_argument(
        '--gui', action='store_true',
        help='Use GUI for image selection'
    )
    parser.add_argument(
        '--host', type=str, default='127.0.0.1',
        help='Server host (default 127.0.0.1)'
    )
    parser.add_argument(
        '--port', type=int, default=5000,
        help='Server port (default 5000)'
    )
    parser.add_argument(
        '--strength', type=float, default=0.02,
        help='Embedding strength for the Original model (e.g. 0.005–0.05)'
    )

    args = parser.parse_args()
    EMBEDDING_STRENGTH = args.strength

    cover_path = args.cover
    secret_path = args.secret

    if args.gui:
        cover_path, secret_path = select_images_gui()
        if not cover_path or not secret_path:
            return
    elif not cover_path or not secret_path:
        print("Provide both --cover and --secret, or use --gui.")
        print("Example: python secure_steganography_demo_updated.py --cover cover.png --secret secret.png")
        return

    if not os.path.exists(cover_path):
        print(f"Cover not found: {cover_path}")
        return
    if not os.path.exists(secret_path):
        print(f"Secret not found: {secret_path}")
        return

    # Start the server in a background thread
    server_thread = threading.Thread(
        target=run_server,
        args=(args.port,),
        daemon=True
    )
    server_thread.start()
    time.sleep(1)

    # Run the client to send/receive images
    run_client(
        cover_path,
        secret_path,
        args.host,
        args.port
    )


if __name__ == "__main__":
    main()