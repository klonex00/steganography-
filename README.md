# Secure Image Steganography System

A secure image steganography framework using deep learning models and encrypted communication. Users can upload a cover image and a secret image to hide one inside the other, transmit it securely, and retrieve it with minimal loss using CNN and UNet models.

---

## 🔍 Overview

- ML-based image steganography
- Secure image transmission using TLS and symmetric encryption
- Streamlit drag‑and‑drop interface
- PSNR and SSIM evaluation for image quality
- Two models: CNN-based and UNet (UNet accuracy lower due to hardware limits)

---

## 🧠 Architecture

- **Client**: Uploads images via Streamlit UI
- **TLS Handshake**: Uses public-key encryption to establish trust
- **Symmetric Encryption**: Uses Fernet (shared key) for image data
- **Server**:
  - Receives and decrypts images
  - Feeds them to ML model (CNN or UNet)
  - Encrypts output and sends back
- **Client**: Decrypts and displays stego + revealed image

---

## ⚙️ Installation

```bash
git clone https://github.com/klonex00/steganography-.git
cd steganography-
pip install -r requirements.txt
