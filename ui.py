import sys
import subprocess
from pathlib import Path

import streamlit as st

# --- Page setup ---
st.set_page_config(
    page_title="Machine Learning-Based Image Steganography System",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Machine Learning-Based Image Steganography System")

# --- Sidebar ---
st.sidebar.title("🔐 Configuration")
models = st.sidebar.multiselect(
    "Models to Run", ["unet", "original"], default=["original"]
)

st.sidebar.markdown(r"""
**Key Points**  
- **UNetSteganoModel**: High-capacity U-Net with skip connections.  
- **OriginalSteganoModel**: Lightweight autoencoder for edge devices.  
- **Loss**: α·MSE + β·Perceptual (α=0.8, β=0.2).  
- **Data**: 128×128 image pairs (CIFAR-10 + custom).  
- **Security**: TLS-wrapped socket channel.
""")

# --- Tabs ---
tabs = st.tabs(["Overview", "Demo"])

with tabs[0]:
    st.header("Overview")
    st.markdown(r"""
A machine-learning framework embeds a secret image into a cover image, then extracts it after a secure TLS transmission:

- **UNetSteganoModel**:  
  - Pros: 0.4–0.5 bpp capacity, multi-scale fidelity  
  - Cons: ≥4 GB GPU

- **OriginalSteganoModel**:  
  - Pros: <1 M params, runs on 2 GB VRAM  
  - Cons: 0.15–0.2 bpp capacity

Upload images in the **Demo** tab to see side-by-side cover → stego → revealed outputs.
""")
    st.markdown("---")

with tabs[1]:
    st.header("Interactive Demo")
    cover = st.file_uploader("Cover Image", type=["png","jpg","jpeg"])
    secret = st.file_uploader("Secret Image", type=["png","jpg","jpeg"])

    if cover: st.image(cover, caption="Cover", width=180)
    if secret: st.image(secret, caption="Secret", width=180)

    if st.button("Start"):
        if not cover or not secret:
            st.error("Upload both images first.")
        else:
            # write uploads to a temp folder next to this script
            BASE_DIR = Path(__file__).resolve().parent
            tmp = BASE_DIR / ".tmp"
            tmp.mkdir(exist_ok=True)
            cpath = tmp / "cover.png"
            spath = tmp / "secret.png"
            cpath.write_bytes(cover.getbuffer())
            spath.write_bytes(secret.getbuffer())

            # build command
            cmd = [
                sys.executable,
                str(BASE_DIR / "secure_steganography_demo_updated.py"),
                "--cover", str(cpath),
                "--secret", str(spath)
            ]

            # run it
            with st.spinner("Running…"):
                proc = subprocess.run(cmd, cwd=str(BASE_DIR), capture_output=True)

            out = proc.stdout.decode() or proc.stderr.decode()
            st.text_area("Console", out, height=200)

            if proc.returncode != 0:
                st.error("Process failed.")
            else:
                st.success("Done!")
                # for each selected model, display its stego & revealed
                for m in models:
                    stego = BASE_DIR / f"{m}_stego.png"
                    rev   = BASE_DIR / f"{m}_revealed.png"
                    if stego.exists() and rev.exists():
                        st.subheader(m.upper())
                        c1, c2 = st.columns(2)
                        c1.image(str(stego), caption="Stego",    width=180)
                        c2.image(str(rev),   caption="Revealed", width=180)

                st.markdown("---")
                st.subheader("Comparison")
                cols = st.columns(1 + len(models))
                cols[0].image(str(cpath), caption="Cover", width=150)
                for i, m in enumerate(models, start=1):
                    cols[i].image(
                        str(BASE_DIR / f"{m}_stego.png"),
                        caption=f"{m.capitalize()} Stego",
                        width=150
                    )
                st.caption("Metrics (PSNR/SSIM) are printed above.")