"""
SHEMATIC Neural Resolver Server  v2
====================================
Full PyTorch neural network with auto-install of dependencies.

Run:  python shematic_neural.py
"""

import sys, os, subprocess, importlib

REQUIRED = [
    ("torch",  "torch", None),
    ("numpy",  "numpy", None),
]

def _ensure(pkg_pip, pkg_import, extra_args=None):
    try:
        importlib.import_module(pkg_import)
        return True
    except ImportError:
        print(f"[setup] installing {pkg_pip}...")
        cmd = [sys.executable, "-m", "pip", "install", "--user", "--upgrade", pkg_pip]
        if extra_args:
            cmd += extra_args
        try:
            subprocess.check_call(cmd)
            importlib.invalidate_caches()
            importlib.import_module(pkg_import)
            print(f"[setup] {pkg_pip} installed.")
            return True
        except Exception as e:
            print(f"[setup] FAILED to install {pkg_pip}: {e}")
            return False

print("=" * 60)
print(" SHEMATIC Neural Resolver - bootstrapping dependencies")
print("=" * 60)
for pip_name, import_name, args in REQUIRED:
    if not _ensure(pip_name, import_name, args):
        print(f"[fatal] required package missing: {pip_name}")
        sys.exit(1)

import json, time, math, random, logging, threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread, Lock
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

HOST = "127.0.0.1"
PORT = 8080

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

DATA_FILE   = os.path.join(MODEL_DIR, "players.json")
MODEL_FILE  = os.path.join(MODEL_DIR, "shematic_model.pt")
REPLAY_FILE = os.path.join(MODEL_DIR, "replay.pt")
METRICS_FILE = os.path.join(MODEL_DIR, "metrics.json")
LOG_FILE    = os.path.join(BASE_DIR, "shematic_server.log")
AUTO_SAVE_INTERVAL = 30

NN_IN_FEATURES = 64
NN_HIDDEN      = [256, 192, 128, 96, 64, 48, 32, 24]
NN_DROPOUT     = 0.10

REPLAY_CAP = 80000
BATCH_SIZE = 64
TRAIN_EVERY = 4
LR_BASE = 5e-4
WD = 2e-5
GRAD_CLIP = 5.0

logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger("shematic")
console = logging.StreamHandler(sys.stdout)
console.setFormatter(logging.Formatter('%(asctime)s %(message)s', datefmt='%H:%M:%S'))
log.addHandler(console)

state_lock = Lock()
player_records = {}
replay_buffer = deque(maxlen=REPLAY_CAP)
prediction_cache = {}  # Кэш для быстрых ответов
cache_lock = Lock()
metrics = {
    "total_requests": 0,
    "total_feedback": 0,
    "total_hits": 0,
    "total_misses": 0,
    "train_steps": 0,
    "avg_loss": 0.0,
    "current_lr": LR_BASE,
    "cache_hits": 0,
    "cache_misses": 0,
}
start_time = time.time()
last_save_time = time.time()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Torch device: {device}")
log.info(f"Torch version: {torch.__version__}")
if torch.cuda.is_available():
    log.info(f"CUDA device: {torch.cuda.get_device_name(0)}")

class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.10):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.norm1   = nn.LayerNorm(out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.norm2   = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        if in_dim != out_dim:
            self.shortcut = nn.Linear(in_dim, out_dim)
        else:
            self.shortcut = nn.Identity()
        self.act = nn.LeakyReLU(0.05)

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.act(self.norm1(self.linear1(x)))
        out = self.dropout(out)
        out = self.norm2(self.linear2(out))
        out = out + identity
        return self.act(out)


class ShematicNet(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        prev = NN_IN_FEATURES
        for sz in NN_HIDDEN:
            layers.append(ResidualBlock(prev, sz, NN_DROPOUT))
            prev = sz
        self.trunk = nn.Sequential(*layers)
        self.head_side  = nn.Linear(prev, 1)
        self.head_conf  = nn.Linear(prev, 1)
        self.head_pitch = nn.Linear(prev, 1)
        self.head_angle = nn.Linear(prev, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.05, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.trunk(x)
        return {
            "side":  torch.sigmoid(self.head_side(h)),
            "conf":  torch.sigmoid(self.head_conf(h)),
            "pitch": torch.tanh(self.head_pitch(h)),
            "angle": torch.tanh(self.head_angle(h)),
        }


model = ShematicNet().to(device)
optimizer = optim.AdamW(model.parameters(), lr=LR_BASE, weight_decay=WD)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1000, T_mult=2, eta_min=1e-5)

param_count = sum(p.numel() for p in model.parameters())
log.info(f"Model parameters: {param_count:,}")


def save_all():
    global last_save_time
    with state_lock:
        try:
            with open(DATA_FILE, "w") as f:
                json.dump(player_records, f)
            torch.save({
                'model': model.state_dict(),
                'optim': optimizer.state_dict(),
                'sched': scheduler.state_dict(),
                'metrics': metrics,
            }, MODEL_FILE)
            if len(replay_buffer) > 0:
                samples = list(replay_buffer)[-20000:]
                torch.save({'samples': samples}, REPLAY_FILE)
            with open(METRICS_FILE, "w") as f:
                json.dump(metrics, f, indent=2)
            last_save_time = time.time()
            log.info(f"Saved. players={len(player_records)} replay={len(replay_buffer)} steps={metrics['train_steps']}")
        except Exception as e:
            log.error(f"Save failed: {e}")


def load_all():
    global player_records, replay_buffer, metrics
    try:
        if os.path.exists(DATA_FILE):
            with open(DATA_FILE) as f:
                player_records = json.load(f)
            log.info(f"Loaded {len(player_records)} player records")
        if os.path.exists(MODEL_FILE):
            ckpt = torch.load(MODEL_FILE, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optim'])
            if 'sched' in ckpt:
                scheduler.load_state_dict(ckpt['sched'])
            if 'metrics' in ckpt:
                metrics.update(ckpt['metrics'])
            log.info(f"Loaded model (train_steps={metrics['train_steps']})")
        if os.path.exists(REPLAY_FILE):
            rep = torch.load(REPLAY_FILE, map_location='cpu', weights_only=False)
            for s in rep.get('samples', []):
                replay_buffer.append(s)
            log.info(f"Loaded {len(replay_buffer)} replay samples")
    except Exception as e:
        log.error(f"Load failed: {e}")


def auto_save_worker():
    while True:
        time.sleep(5)
        if time.time() - last_save_time >= AUTO_SAVE_INTERVAL:
            save_all()


def get_cache_key(features, context):
    """Generate cache key from features and context"""
    import hashlib
    import json
    data = json.dumps({"features": features, "context": context}, sort_keys=True)
    return hashlib.md5(data.encode()).hexdigest()

def get_cached_prediction(cache_key):
    """Get prediction from cache if available and fresh"""
    with cache_lock:
        if cache_key in prediction_cache:
            cached = prediction_cache[cache_key]
            if time.time() - cached["timestamp"] < 0.15:  # 150ms cache TTL
                metrics["cache_hits"] += 1
                return cached["result"]
            else:
                del prediction_cache[cache_key]
    return None

def cache_prediction(cache_key, result):
    """Cache prediction result"""
    with cache_lock:
        prediction_cache[cache_key] = {
            "result": result,
            "timestamp": time.time()
        }
        if len(prediction_cache) > 1000:  # Limit cache size
            oldest = min(prediction_cache.keys(), key=lambda k: prediction_cache[k]["timestamp"])
            del prediction_cache[oldest]

def add_replay(features, hit, side, pitch, angle, weight=1.0):
    if len(features) != NN_IN_FEATURES:
        return
    replay_buffer.append({
        'f': list(features),
        'hit': bool(hit),
        'side': int(side),
        'pitch': float(pitch),
        'angle': float(angle),
        'w': float(weight),
        't': time.time(),
    })


def train_step():
    if len(replay_buffer) < BATCH_SIZE * 2:
        return None
    model.train()

    indices = np.random.choice(len(replay_buffer), size=BATCH_SIZE, replace=False)
    batch = [replay_buffer[i] for i in indices]

    X = torch.tensor([b['f'] for b in batch], dtype=torch.float32, device=device)
    y_side = torch.tensor([
        [1.0 if b['hit'] and b['side'] > 0 else
         0.0 if b['hit'] and b['side'] < 0 else
         0.0 if not b['hit'] and b['side'] > 0 else
         1.0]
        for b in batch
    ], dtype=torch.float32, device=device)
    y_conf  = torch.tensor([[1.0 if b['hit'] else 0.0] for b in batch], dtype=torch.float32, device=device)
    y_pitch = torch.tensor([[b['pitch'] / 89.0] for b in batch], dtype=torch.float32, device=device)
    y_angle = torch.tensor([[b['angle'] / 58.0] for b in batch], dtype=torch.float32, device=device)
    W       = torch.tensor([b['w'] for b in batch], dtype=torch.float32, device=device).unsqueeze(-1)

    out = model(X)
    bce_side = -(y_side * torch.log(out['side'] + 1e-7) +
                 (1 - y_side) * torch.log(1 - out['side'] + 1e-7))
    p_t = torch.where(y_side > 0.5, out['side'], 1 - out['side'])
    focal_w = (1 - p_t) ** 2
    loss_side = ((bce_side * (1.0 + 1.5 * focal_w)) * W).mean()
    loss_conf = F.binary_cross_entropy(out['conf'], y_conf, weight=W)
    loss_pitch = (F.mse_loss(out['pitch'], y_pitch, reduction='none') * W).mean()
    loss_angle = (F.smooth_l1_loss(out['angle'], y_angle, reduction='none') * W).mean()

    loss = loss_side + 0.5 * loss_conf + 0.3 * loss_pitch + 0.4 * loss_angle

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    optimizer.step()
    scheduler.step()

    metrics['train_steps'] += 1
    a = 0.98
    metrics['avg_loss'] = a * metrics['avg_loss'] + (1 - a) * float(loss.item()) if metrics['avg_loss'] > 0 else float(loss.item())
    metrics['current_lr'] = float(optimizer.param_groups[0]['lr'])
    return float(loss.item())


def statistical_resolve(rec, ctx):
    history = rec.get("side_history", [])
    miss = rec.get("missed_shots", 0)
    if not history:
        if ctx.get("real_side") and ctx.get("side_conf", 0) > 0.55:
            return ctx["real_side"], ctx.get("max_desync", 58) * 0.85 * ctx["real_side"], 0.70, "STAT-HINT"
        return 0, 0, 0, "STAT-NONE"

    left = sum(1 for s in history if s == -1)
    right = sum(1 for s in history if s == 1)
    total = left + right
    if total == 0:
        return 0, 0, 0, "STAT-EMPTY"

    flips = sum(1 for i in range(1, len(history)) if history[i] != history[i-1])
    stability = 1 - flips / max(1, len(history) - 1)

    wsum = 0
    weights = [0.4, 0.6, 0.8, 1.0]
    recent = history[-4:]
    for i, s in enumerate(recent):
        if s != 0:
            wsum += s * weights[-(len(recent) - i)]

    if miss >= 3:
        side = -1 if wsum > 0 else 1
    else:
        side = 1 if wsum > 0 else (-1 if wsum < 0 else (1 if right > left else -1))

    dominant = max(left, right) / total
    conf = min(0.95, 0.40 + dominant * 0.45 * stability)
    max_d = ctx.get("max_desync", 58)
    return side, side * max_d * conf, conf, "STAT"


def _read_json(handler):
    n = int(handler.headers.get("Content-Length", 0))
    if n == 0:
        return None
    body = handler.rfile.read(n).decode("utf-8", errors="ignore")
    try:
        return json.loads(body)
    except Exception:
        return None


def _send_json(handler, code, obj):
    payload = json.dumps(obj).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(payload)))
    handler.end_headers()
    handler.wfile.write(payload)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        log.debug(fmt % args)

    def do_GET(self):
        if self.path.startswith("/status"):
            _send_json(self, 200, {
                "status": "running",
                "uptime": time.time() - start_time,
                "players": len(player_records),
                "requests": metrics["total_requests"],
                "feedback": metrics["total_feedback"],
                "hits": metrics["total_hits"],
                "misses": metrics["total_misses"],
                "train_steps": metrics["train_steps"],
                "avg_loss": metrics["avg_loss"],
                "current_lr": metrics["current_lr"],
                "replay": len(replay_buffer),
                "backend": "pytorch",
                "device": str(device),
                "params": param_count,
            })
            return
        if self.path.startswith("/metrics"):
            _send_json(self, 200, metrics)
            return
        if self.path.startswith("/export"):
            w = {k: v.cpu().numpy().tolist() for k, v in model.state_dict().items()}
            _send_json(self, 200, w)
            return
        _send_json(self, 404, {"error": "not found"})

    def do_POST(self):
        if self.path.startswith("/feedback"):
            data = _read_json(self)
            if not data:
                _send_json(self, 400, {"error": "invalid json"}); return
            sid  = str(data.get("steam_id", ""))
            feat = data.get("features", [])
            hit  = bool(data.get("hit", False))
            ang  = float(data.get("angle_used", 0))
            ctx  = data.get("context", {})

            if len(feat) != NN_IN_FEATURES:
                _send_json(self, 400, {"error": f"need {NN_IN_FEATURES} features"})
                return

            side = 1 if ang > 0 else -1
            pitch = float(ctx.get("pitch", 0))

            with state_lock:
                rec = player_records.setdefault(sid, {
                    "side_history": [], "hit_angles": [], "miss_angles": [],
                    "missed_shots": 0, "hit_shots": 0, "last_updated": time.time(),
                })
                rec["last_updated"] = time.time()
                if hit:
                    rec["hit_shots"]  = rec.get("hit_shots", 0) + 1
                    rec["missed_shots"] = max(0, rec["missed_shots"] - 1)
                    rec["hit_angles"].append(ang); rec["hit_angles"] = rec["hit_angles"][-30:]
                    rec["side_history"].append(side)
                    metrics["total_hits"] += 1
                else:
                    rec["missed_shots"] = rec.get("missed_shots", 0) + 1
                    rec["miss_angles"].append(ang); rec["miss_angles"] = rec["miss_angles"][-30:]
                    rec["side_history"].append(-side)
                    metrics["total_misses"] += 1
                rec["side_history"] = rec["side_history"][-40:]

                w = 1.5 if hit else 1.0
                add_replay(feat, hit, side, pitch, ang, w)
                metrics["total_feedback"] += 1

            if metrics["total_feedback"] % TRAIN_EVERY == 0:
                loss = train_step()
                if loss is not None and metrics["train_steps"] % 50 == 0:
                    log.info(f"step={metrics['train_steps']} loss={loss:.4f} lr={metrics['current_lr']:.6f}")

            _send_json(self, 200, {"ok": True})
            return

        data = _read_json(self)
        if not data:
            _send_json(self, 400, {"error": "invalid json"}); return
        sid  = str(data.get("steam_id", ""))
        feat = data.get("features", [])
        ctx  = data.get("context", {})

        if not sid or len(feat) != NN_IN_FEATURES:
            _send_json(self, 400, {"error": f"need steam_id + {NN_IN_FEATURES} features"})
            return

        # Check cache first for faster response
        cache_key = get_cache_key(feat, ctx)
        cached_result = get_cached_prediction(cache_key)
        if cached_result:
            _send_json(self, 200, cached_result)
            return

        with state_lock:
            rec = player_records.setdefault(sid, {
                "side_history": [], "hit_angles": [], "miss_angles": [],
                "missed_shots": 0, "hit_shots": 0, "last_updated": time.time(),
            })
            stat_side, stat_angle, stat_conf, stat_method = statistical_resolve(rec, ctx)
            miss_streak = rec.get("missed_shots", 0)

        model.eval()
        with torch.no_grad():
            x = torch.tensor(feat, dtype=torch.float32, device=device).unsqueeze(0)
            out = model(x)
            side_p  = float(out['side'].item())
            conf_p  = float(out['conf'].item())
            pitch_p = float(out['pitch'].item())
            angle_p = float(out['angle'].item())

        nn_side = 1 if side_p > 0.5 else -1
        nn_conf = conf_p * abs(side_p - 0.5) * 2
        max_d = ctx.get("max_desync", 58)
        nn_angle = angle_p * max_d
        if abs(nn_angle) < max_d * 0.30:
            nn_angle = nn_side * max_d * max(0.45, conf_p)
        nn_pitch = int(89 if pitch_p > 0.7 else (-89 if pitch_p < -0.7 else 0))
        if nn_pitch == 0:
            nn_pitch = None

        if metrics["total_feedback"] > 80 and nn_conf > 0.45:
            final_side, final_angle, final_conf, method = nn_side, nn_angle, nn_conf, "NN"
        elif stat_conf > 0.60:
            final_side, final_angle, final_conf, method = stat_side, stat_angle, stat_conf, stat_method
        else:
            if nn_side == stat_side and nn_side != 0:
                final_side  = nn_side
                final_angle = nn_angle * 0.6 + stat_angle * 0.4
                final_conf  = max(nn_conf, stat_conf) * 1.05
                method = "BLEND"
            elif nn_conf > stat_conf:
                final_side, final_angle, final_conf, method = nn_side, nn_angle, nn_conf, "NN-W"
            else:
                final_side, final_angle, final_conf, method = stat_side, stat_angle, stat_conf, "STAT-W"

        stage = 0
        if miss_streak >= 3:
            stage = (miss_streak - 2) % 5
            stages = [1.0, -1.0, 0.6, -0.6, 0.0]
            mult = stages[stage]
            final_angle = max_d * mult * (1 if final_side >= 0 else -1)
            method = f"BRUTE-{stage}"
            final_conf = max(0.50, final_conf * 0.85)

        pitch_force = nn_pitch
        if ctx.get("defensive"):
            p = ctx.get("pitch", 0)
            if p > 60: pitch_force = 89
            elif p < -60: pitch_force = -89

        metrics["total_requests"] += 1
        metrics["cache_misses"] += 1
        if metrics["total_requests"] % 250 == 0:
            log.info(f"req#{metrics['total_requests']} sid={sid[-4:]} m={method} side={final_side} conf={final_conf:.2f} cache_hits={metrics['cache_hits']} cache_misses={metrics['cache_misses']}")

        result = {
            "side":        int(final_side),
            "angle":       round(float(final_angle), 2),
            "pitch_force": pitch_force,
            "confidence":  round(float(final_conf), 3),
            "method":      method,
            "stage":       stage,
        }
        
        # Cache the result for future requests
        cache_prediction(cache_key, result)
        
        _send_json(self, 200, result)


def main():
    load_all()
    Thread(target=auto_save_worker, daemon=True).start()
    try:
        srv = HTTPServer((HOST, PORT), Handler)
        log.info("=" * 60)
        log.info(f" SHEMATIC Neural Resolver v2 - PyTorch")
        log.info(f" Listening: http://{HOST}:{PORT}")
        log.info(f" Device: {device}  Params: {param_count:,}")
        log.info(f" Models dir: {MODEL_DIR}")
        log.info("=" * 60)
        srv.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutdown - saving state...")
        save_all()
        log.info("Done.")
    except Exception as e:
        log.exception(f"Fatal: {e}")


if __name__ == "__main__":
    main()
