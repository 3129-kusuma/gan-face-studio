
import json, os
import numpy as np
import streamlit as st
import torch, torch.nn as nn
import cv2
from PIL import Image
from huggingface_hub import hf_hub_download

MODEL_REPO = "kusuma4546/celeba-dcgan-faces"
HF_TOKEN = os.environ.get("HF_TOKEN")
IMG = 64

st.set_page_config(page_title="GAN Face Studio", page_icon="🧑‍🎨", layout="wide")
st.markdown("""
<style>
.stApp {background: radial-gradient(circle at 15% 0%, #261b3b 0%, #0b0d1a 60%);}
h1.hero {font-size: 3rem; font-weight: 800; margin-bottom: 0;
  background: linear-gradient(90deg,#f45d9b,#7f5af0,#2cb67d);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;}
.sub {color:#a7a9be; font-size:1.1rem; margin-bottom:1.2rem;}
.card {background: rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.1);
  border-radius:16px; padding:8px;}
div.stButton > button {background: linear-gradient(90deg,#f45d9b,#7f5af0); color:white;
  border:0; border-radius:12px; padding:0.7rem 1.4rem; font-weight:700; width:100%;}
div.stButton > button:hover {filter:brightness(1.15);}
</style>
""", unsafe_allow_html=True)

class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super().__init__()
        def up(i, o, k=4, s=2, p=1):
            return [nn.ConvTranspose2d(i, o, k, s, p, bias=False), nn.BatchNorm2d(o), nn.ReLU(True)]
        self.net = nn.Sequential(*up(z_dim, 512, 4, 1, 0), *up(512, 256), *up(256, 128), *up(128, 64),
                                 nn.ConvTranspose2d(64, 3, 4, 2, 1), nn.Tanh())
    def forward(self, z):
        return self.net(z.view(z.size(0), -1, 1, 1))

@st.cache_resource(show_spinner="Loading GAN from Hugging Face Hub...")
def load_generator():
    cfg = json.load(open(hf_hub_download(MODEL_REPO, "config.json", token=HF_TOKEN)))
    g = Generator(cfg["z_dim"])
    g.load_state_dict(torch.load(hf_hub_download(MODEL_REPO, "generator.pt", token=HF_TOKEN), map_location="cpu"))
    g.eval()
    for p in g.parameters():
        p.requires_grad_(False)
    return g, cfg["z_dim"]

G, Z_DIM = load_generator()
def _load_cascade():
       name = "haarcascade_frontalface_default.xml"
       paths = []
       try:
           paths.append(cv2.data.haarcascades + name)
       except Exception:
           pass
       paths.append(os.path.join(os.path.dirname(cv2.__file__), "data", name))
       for p in paths:
           if os.path.exists(p):
               c = cv2.CascadeClassifier(p)
               if not c.empty():
                   return c
       return None

   cascade = _load_cascade()

def crop_face(img, margin=1.8):
    img = img.convert("RGB")
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    found = cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40)) if cascade is not None else []
    h, w = gray.shape
    if len(found):
        x, y, fw, fh = max(found, key=lambda f: f[2] * f[3])
        cx, cy, s = x + fw / 2, y + fh / 2, max(fw, fh) * margin
        detected = True
    else:
        cx, cy, s = w / 2, h / 2, min(w, h)
        detected = False
    r = s / 2
    return img.crop((int(cx - r), int(cy - r), int(cx + r), int(cy + r))).resize((IMG, IMG), Image.LANCZOS), detected

def pil_to_tensor(im):
    return (torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 127.5 - 1).unsqueeze(0)

def to_images(z, size=256):
    with torch.no_grad():
        x = G(z).permute(0, 2, 3, 1).numpy()
    x = ((x + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    return [Image.fromarray(i).resize((size, size), Image.LANCZOS) for i in x]

def latent(seed, scale=1.0):
    return torch.randn(1, Z_DIM, generator=torch.Generator().manual_seed(int(seed))) * scale

def invert(target, steps=200, restarts=6, progress=None):
    """GAN inversion: optimise several random z so that G(z) matches the target photo."""
    t = target.repeat(restarts, 1, 1, 1)
    z = torch.randn(restarts, Z_DIM).requires_grad_(True)
    opt = torch.optim.Adam([z], lr=0.05)
    for s in range(steps):
        opt.zero_grad()
        per = ((G(z) - t) ** 2).mean(dim=(1, 2, 3))
        (per + 1e-3 * (z ** 2).mean(dim=1)).sum().backward()
        opt.step()
        if progress:
            progress.progress((s + 1) / steps)
    with torch.no_grad():
        per = ((G(z) - t) ** 2).mean(dim=(1, 2, 3))
    best = int(per.argmin())
    return z[best:best + 1].detach(), float(per[best])

st.markdown('<h1 class="hero">🧑‍🎨 GAN Face Studio</h1>', unsafe_allow_html=True)
st.markdown('<div class="sub">Faces of people who do not exist - plus a tool to project your own photo into the GAN.</div>',
            unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["✨ Generate faces", "🔀 Morph two faces", "📷 Upload a photo"])

with tab1:
    c1, c2, c3 = st.columns(3)
    n = c1.slider("Number of faces", 1, 12, 8)
    scale = c2.slider("Creativity (noise scale)", 0.5, 1.5, 1.0, 0.1)
    seed = c3.number_input("Seed", 0, 10**6, 42)
    if st.button("🎲 Generate new faces") or "first" not in st.session_state:
        st.session_state["first"] = True
        rng = np.random.default_rng(int(seed))
        z = torch.tensor(rng.standard_normal((n, Z_DIM)), dtype=torch.float32) * scale
        cols = st.columns(4)
        for i, im in enumerate(to_images(z)):
            with cols[i % 4]:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.image(im, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    a, b = st.columns(2)
    sa = a.number_input("Seed of face A", 0, 10**6, 1)
    sb = b.number_input("Seed of face B", 0, 10**6, 2)
    steps = st.slider("Frames", 3, 9, 7)
    if st.button("🔀 Morph"):
        za, zb = latent(sa), latent(sb)
        z = torch.cat([za * (1 - t) + zb * t for t in np.linspace(0, 1, steps)])
        for c, im in zip(st.columns(steps), to_images(z, 160)):
            c.image(im, use_container_width=True)

with tab3:
    st.caption("🔒 Your photo is processed in memory for this session only and is not stored. "
               "Only upload photos you have the right to use.")
    up = st.file_uploader("Upload a face photo (JPG/PNG)", type=["jpg", "jpeg", "png"])
    c1, c2 = st.columns(2)
    steps_i = c1.slider("Optimisation steps (more = closer match, slower)", 50, 500, 200, 50)
    restarts = c2.slider("Random restarts", 2, 10, 6)

    if up is not None:
        face, detected = crop_face(Image.open(up))
        if not detected:
            st.warning("No face detected - used a centre crop. Try a clearer, front-facing photo.")
        if st.button("🔍 Find this face in the GAN"):
            bar = st.progress(0.0, text="Searching the latent space...")
            z_user, err = invert(pil_to_tensor(face), steps_i, restarts, bar)
            bar.empty()
            st.session_state["z_user"], st.session_state["err"] = z_user, err
            st.session_state["face"] = face

    if "z_user" in st.session_state:
        z_user = st.session_state["z_user"]
        r1, r2 = st.columns(2)
        r1.image(st.session_state["face"].resize((256, 256), Image.LANCZOS), caption="Your photo (cropped)")
        r2.image(to_images(z_user)[0], caption=f"GAN reconstruction (error {st.session_state['err']:.3f})")
        st.caption("The GAN can only produce faces it has learned, so the match is an approximation - "
                   "closer when it was fine-tuned on similar photos.")

        st.subheader("🔀 Blend with another face")
        mix_seed = st.number_input("Seed of the other face", 0, 10**6, 7, key="mixseed")
        t = st.slider("Blend (0 = your reconstruction, 1 = other face)", 0.0, 1.0, 0.5, 0.05)
        zb = latent(mix_seed)
        zz = torch.cat([z_user * (1 - k) + zb * k for k in [0, t, 1]])
        for c, im, cap in zip(st.columns(3), to_images(zz, 200), ["You", f"Blend {t:.2f}", "Other"]):
            c.image(im, caption=cap, use_container_width=True)

        st.subheader("🎲 Variations")
        spread = st.slider("How different", 0.1, 1.0, 0.4, 0.05)
        if st.button("Make variations"):
            zs = z_user + spread * torch.randn(8, Z_DIM)
            cols = st.columns(4)
            for i, im in enumerate(to_images(zs, 200)):
                cols[i % 4].image(im, use_container_width=True)

with st.expander("How does it work?"):
    st.write("A **Generator** turns random noise into a face; a **Discriminator** tries to tell it from real photos. "
             "**GAN inversion** runs the process backwards: it tunes the noise until the generated face looks like "
             "your photo. Blending and variations then move around that noise vector.")
