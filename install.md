# Ripon Computer – AI Photo Studio (v3.1) – Install

Windows (CMD ba PowerShell). Python 3.10+ lagbe (app er jonno). Internet lagbe.

---

## 1. App chalate je package lagbe (ekbar)

```
pip install PySide6 opencv-python numpy pillow rembg onnxruntime
```

Chalate:

```
python ripon_computer_v3_1.py
```

### Optional (na dileo app cholbe, kintu feature kome jabe)

| Feature | Command |
|---|---|
| GPU diye background remove (NVIDIA) | `pip uninstall onnxruntime` then `pip install onnxruntime-gpu` |
| Bhalo face / chokh detect (InsightFace, non-commercial) | `pip install insightface` |
| Face restore / upscale | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121` then `pip install gfpgan realesrgan basicsr` |

Model gulo download korte: app er **Help → AI Models** e row select kore **Download selected**.

Dress / suit PNG: app er pashe `dress` folder e transparent PNG rakhun (sub-folder: Shirt, Suit, Tie, Coat).

---

## 2. AI Pose Straighten (LivePortrait) – manual install

Eta app er **Help → AI Models → AI Pose Straighten (LivePortrait)** theke auto-install-o hoy.
Auto-install fail korle nicher command gulo ek ek kore chalan.

> LivePortrait **Python 3.10** chay. Apnar main Python notun hole `uv` diye alada Python 3.10 environment (`venv`) banate hobe. Eta main Python ke bhange na.
>
> Licence: LivePortrait er face detector (InsightFace) shudhu non-commercial use er jonno.

### Step 0 – folder

```
mkdir C:\Users\Ripon\AppData\Local\RiponComputer\ai_pose
cd /d C:\Users\Ripon\AppData\Local\RiponComputer\ai_pose
```

### Step 1 – code download

(`LivePortrait` folder already thakle ei step skip korun.)

```
git clone --depth 1 https://github.com/KwaiVGI/LivePortrait
```

Git na thakle: https://github.com/KwaiVGI/LivePortrait → **Code → Download ZIP** → `ai_pose` folder e `LivePortrait` naam e extract korun.

### Step 2 – private Python 3.10 environment

`uv` Python 3.10 nijei download kore.

```
pip install --upgrade uv
python -m uv venv --python 3.10 venv
```

### Step 3 – package install (torch ityadi, kayek GB)

`--index-strategy unsafe-best-match` dile NVIDIA GPU er torch version pabe.

```
cd LivePortrait
python -m uv pip install --python ..\venv\Scripts\python.exe --index-strategy unsafe-best-match -r requirements.txt
python -m uv pip install --python ..\venv\Scripts\python.exe huggingface_hub
```

### Step 4 – AI model file download (kayek GB)

```
..\venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download as d; d('KwaiVGI/LivePortrait', local_dir='pretrained_weights', ignore_patterns=['*.git*','README.md','docs/*','liveportrait_animals/*'])"
```

### Step 5 – check

Help text ashle install thik ache.

```
..\venv\Scripts\python.exe inference.py --help
```

### Step 6 – app e connect korun

1. `python ripon_computer_v3_1.py` chalan.
2. **PHOTO EDIT** tab → **🤖 AI Straighten Pose (LivePortrait)**.
3. "Install automatically?" prompt e **No** chapun.
4. Ei folder select korun:
   `C:\Users\Ripon\AppData\Local\RiponComputer\ai_pose\LivePortrait`
5. Ekta **soja, samne takano reference photo** select korun (je kono manush, mukh bondho). AI shudhu oi photo er matha er pose copy kore, customer er chehara thake. Ekbar select korle mone thake.

`LivePortrait` folder er pashe `venv` folder thakle app nijei oi Python use kore.

### Tips

- Age **Remove Background** korun, tarpor pose straighten korun.
- Reference photo te mukh bondho, neutral rakhun, na hole customer er mukhe ektu prabhab porte pare.
- NVIDIA GPU na thakle onek dheere cholbe.
- Reset korte: **AI Tools → Reset LivePortrait folder / reference**.

---

## 3. Somossya hole

| Somossya | Kaaj |
|---|---|
| `numpy` / `pgcc` / `meson` error | Main Python e install korben na. Upore Step 2 er `venv` (Python 3.10) e install korun. |
| `uv` command pay na | `python -m uv ...` formate chalan (upore dewa ache). |
| `huggingface` download beche jay | Step 4 command abar chalan, ja download hoyeche ta rekhe baki ta nibe. |
| App bole "inference.py or pretrained_weights not found" | Step 1 ar Step 4 sesh hoyeche kina dekhun. Folder `LivePortrait` select korte hobe. |
| Kono error | Puro error text ta copy kore pathan. |
