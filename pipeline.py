import cv2
import torch
import numpy as np
import warnings
from collections import deque, defaultdict
from scipy.spatial.distance import cdist
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights
import supervision as sv
from ultralytics import YOLO

try:
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    SAHI_AVAILABLE = True
except ImportError:
    AutoDetectionModel = None
    get_sliced_prediction = None
    SAHI_AVAILABLE = False

from model import TheftDetectionLSTM

warnings.filterwarnings("ignore")

POSE_MODEL_NAME = 'yolo11x-pose.pt'

SEG_MODEL_NAME  = 'yolo11x-seg.pt'

# Object detection model for high-value items (phones, laptops, bags, etc.)
OBJ_MODEL_NAME  = 'yolo11x.pt'

# COCO classes we care about as potential theft targets.
# Ultralytics COCO IDs: 67=cell phone, 63=laptop, 26=handbag, 28=suitcase,
# 73=book, 74=clock, 64=mouse, 65=remote, 66=keyboard, 76=scissors,
# plus common small store objects like bottle/cup/food/vase/toothbrush.
THEFT_TARGET_CLASSES = {
    67, 63, 26, 28, 73, 74, 64, 65, 66, 76,
    39, 40, 41, 42, 43, 44, 45,
    46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
    75, 78, 79,
}
TARGET_CLASS_NAMES = {
    "cell phone", "laptop", "handbag", "suitcase", "book", "clock",
    "mouse", "remote", "keyboard", "scissors", "bottle", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "vase", "hair drier", "toothbrush",
}
COCO_NAMES = {
    26: "handbag", 28: "suitcase", 39: "bottle", 40: "wine glass",
    41: "cup", 42: "fork", 43: "knife", 44: "spoon", 45: "bowl",
    46: "banana", 47: "apple", 48: "sandwich", 49: "orange",
    50: "broccoli", 51: "carrot", 52: "hot dog", 53: "pizza",
    54: "donut", 55: "cake", 63: "laptop", 64: "mouse",
    65: "remote", 66: "keyboard", 67: "phone", 73: "book",
    74: "clock", 75: "vase", 76: "scissors", 78: "hair drier",
    79: "toothbrush",
}

# SAHI tiled inference improves tiny-object recall on high-resolution shop/store videos.
# Smaller slices see phones and shelf items larger; overlap prevents boundary misses.
USE_SAHI_OBJECT_DETECTION = True
OBJ_CONF = 0.22
SAHI_SLICE_HEIGHT = 1000
SAHI_SLICE_WIDTH = 1000
SAHI_OVERLAP_RATIO = 0.25
SAHI_BATCH_SIZE = 4
SAHI_PERFORM_STANDARD_PRED = True

# How close (normalised by frame width) a wrist must be to an object to count as "touching"
OBJECT_TOUCH_RADIUS = 0.08

# If an object disappears within this many frames after a hand was near it → strong signal
OBJECT_MISSING_PATIENCE = 45

OBJ_INFER_EVERY_N = 2   # run object detector every N frames (balance speed vs accuracy)

USE_BOTSORT = True

USE_HALF       = True
SEG_EVERY_N    = 3
INFER_SIZE     = 640
SKIP_EMBED_EVERY = 2
MAX_DISPLAY_W  = 1280

KP_NOSE           = 0
KP_LEFT_EYE       = 1
KP_RIGHT_EYE      = 2
KP_LEFT_EAR       = 3
KP_RIGHT_EAR      = 4
KP_LEFT_SHOULDER  = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_ELBOW     = 7
KP_RIGHT_ELBOW    = 8
KP_LEFT_WRIST     = 9
KP_RIGHT_WRIST    = 10
KP_LEFT_HIP       = 11
KP_RIGHT_HIP      = 12
KP_LEFT_KNEE      = 13
KP_RIGHT_KNEE     = 14
KP_LEFT_ANKLE     = 15
KP_RIGHT_ANKLE    = 16

SKELETON_CONNECTIONS = [
    (KP_LEFT_SHOULDER,  KP_RIGHT_SHOULDER),
    (KP_LEFT_SHOULDER,  KP_LEFT_ELBOW),
    (KP_LEFT_ELBOW,     KP_LEFT_WRIST),
    (KP_RIGHT_SHOULDER, KP_RIGHT_ELBOW),
    (KP_RIGHT_ELBOW,    KP_RIGHT_WRIST),
    (KP_LEFT_SHOULDER,  KP_LEFT_HIP),
    (KP_RIGHT_SHOULDER, KP_RIGHT_HIP),
    (KP_LEFT_HIP,       KP_RIGHT_HIP),
    (KP_LEFT_HIP,       KP_LEFT_KNEE),
    (KP_LEFT_KNEE,      KP_LEFT_ANKLE),
    (KP_RIGHT_HIP,      KP_RIGHT_KNEE),
    (KP_RIGHT_KNEE,     KP_RIGHT_ANKLE),
    (KP_NOSE,           KP_LEFT_SHOULDER),
    (KP_NOSE,           KP_RIGHT_SHOULDER),
]

def angle_between(a, b, c):
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_a, -1, 1)))

def bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def bbox_area(bbox):
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)

def bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix = max(0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    union = bbox_area(a) + bbox_area(b) - inter + 1e-6
    return inter / union

def loitering_score(pts, frame_count=30):
    if len(pts) < frame_count:
        return 0.0
    recent = np.array(list(pts)[-frame_count:])
    return max(0.0, 1.0 - np.std(recent, axis=0).mean() / 50.0)

class SkeletonAnalyzer:
    def __init__(self):
        self.prev_keypoints = {}
        self.head_direction_history = defaultdict(lambda: deque(maxlen=5))
        self.wrist_height_history = defaultdict(lambda: deque(maxlen=15))
        self.spine_angle_history = defaultdict(lambda: deque(maxlen=10))

    def transfer_state(self, old_id, new_id):
        for store in [
            self.prev_keypoints,
            self.head_direction_history,
            self.wrist_height_history,
            self.spine_angle_history,
        ]:
            if old_id in store:
                store[new_id] = store.pop(old_id)

    def extract(self, person_id, keypoints, bbox, frame_h, frame_w):
        feats = {k: 0.0 for k in [
            "wrist_to_pocket_dist", "elbow_angle_left", "elbow_angle_right",
            "arm_extension", "head_turn_speed", "gaze_direction_change",
            "spine_lean_forward", "wrist_above_table", "wrist_moving_toward_body",
            "grab_motion_score", "both_arms_forward", "wrist_cross_midline",
            "torso_twist", "knee_bend", "shoulder_asymmetry", "step_ready_stance",
            "wrist_speed",
        ]}

        def kp(idx):
            x, y, c = keypoints[idx]
            return (float(x), float(y)) if c > 0.25 else None

        bh = max(bbox[3] - bbox[1], 1e-6)
        bw = max(bbox[2] - bbox[0], 1e-6)

        lhip = kp(11); rhip = kp(12)
        lwrist = kp(9); rwrist = kp(10)
        lelbow = kp(7); relbow = kp(8)
        lsh = kp(5); rsh = kp(6)
        lknee = kp(13); rknee = kp(14)
        lank = kp(15); rank = kp(16)
        nose = kp(0)

        if lhip and rhip:
            hx = (lhip[0] + rhip[0]) / 2
            hy = (lhip[1] + rhip[1]) / 2
            for w in [lwrist, rwrist]:
                if w:
                    feats["wrist_to_pocket_dist"] = max(
                        feats["wrist_to_pocket_dist"],
                        1.0 - min(np.hypot(w[0] - hx, w[1] - hy) / bh, 1.0)
                    )

        if lsh and lelbow and lwrist:
            feats["elbow_angle_left"] = angle_between(lsh, lelbow, lwrist) / 180.0
        if rsh and relbow and rwrist:
            feats["elbow_angle_right"] = angle_between(rsh, relbow, rwrist) / 180.0

        best_ext = 0.0
        for sh, wr in [(lsh, lwrist), (rsh, rwrist)]:
            if sh and wr:
                best_ext = max(best_ext, np.hypot(wr[0] - sh[0], wr[1] - sh[1]) / bh)
        feats["arm_extension"] = min(best_ext, 1.0)

        if lsh and rsh and lhip and rhip:
            sh_cx = (lsh[0] + rsh[0]) / 2
            sh_cy = (lsh[1] + rsh[1]) / 2
            hcx = (lhip[0] + rhip[0]) / 2
            hcy = (lhip[1] + rhip[1]) / 2
            horiz = abs(sh_cx - hcx) / (bw + 1e-6)
            sa = abs(np.degrees(np.arctan2(abs(sh_cx - hcx), abs(sh_cy - hcy) + 1e-6)))
            raw = max(horiz, min(sa / 45.0, 1.0) * 0.5)
            self.spine_angle_history[person_id].append(raw)
            hist = list(self.spine_angle_history[person_id])
            feats["spine_lean_forward"] = np.mean(hist[-4:]) if len(hist) >= 4 else raw * 0.5

        if lknee or rknee:
            ky, cnt = 0.0, 0
            for k in [lknee, rknee]:
                if k:
                    ky += k[1]
                    cnt += 1
            ky /= max(cnt, 1)
            for w in [lwrist, rwrist]:
                if w and w[1] < ky:
                    feats["wrist_above_table"] = max(
                        feats["wrist_above_table"],
                        min((ky - w[1]) / bh, 1.0)
                    )

        self.wrist_height_history[person_id].append(
            (lwrist[1] if lwrist else 0, rwrist[1] if rwrist else 0)
        )
        hw = self.wrist_height_history[person_id]
        if len(hw) >= 4 and rwrist:
            mv = (hw[-1][1] - hw[-4][1]) / (bh + 1e-6)
            feats["wrist_moving_toward_body"] = max(0.0, min(mv * 3, 1.0))

        feats["grab_motion_score"] = min(
            feats["spine_lean_forward"] * 0.35 +
            feats["arm_extension"] * 0.30 +
            feats["wrist_above_table"] * 0.20 +
            feats["wrist_moving_toward_body"] * 0.15,
            1.0
        )

        if lsh and rsh:
            smx = (lsh[0] + rsh[0]) / 2
            lf = (lwrist[0] - smx) / (bw + 1e-6) if lwrist else 0
            rf = (rwrist[0] - smx) / (bw + 1e-6) if rwrist else 0
            if lf > 0.1 and rf > 0.1:
                feats["both_arms_forward"] = min((lf + rf) / 2, 1.0)

        if lhip and rhip and lwrist and rwrist:
            mx = (lhip[0] + rhip[0]) / 2
            cr = 0.0
            if lwrist[0] > mx:
                cr = max(cr, (lwrist[0] - mx) / (bw + 1e-6))
            if rwrist[0] < mx:
                cr = max(cr, (mx - rwrist[0]) / (bw + 1e-6))
            feats["wrist_cross_midline"] = min(cr, 1.0)

        if lsh and rsh and lhip and rhip:
            feats["torso_twist"] = min(abs((lsh[0] - rsh[0]) - (lhip[0] - rhip[0])) / (bw + 1e-6), 1.0)

        if lhip and lknee and lank:
            feats["knee_bend"] = max(feats["knee_bend"], 1.0 - angle_between(lhip, lknee, lank) / 180.0)
        if rhip and rknee and rank:
            feats["knee_bend"] = max(feats["knee_bend"], 1.0 - angle_between(rhip, rknee, rank) / 180.0)

        if lsh and rsh:
            feats["shoulder_asymmetry"] = min(abs(lsh[1] - rsh[1]) / (bh + 1e-6) * 2, 1.0)

        if nose:
            self.head_direction_history[person_id].append(nose)
            hd = self.head_direction_history[person_id]
            if len(hd) >= 2:
                feats["head_turn_speed"] = min(
                    np.hypot(hd[-1][0] - hd[-2][0], hd[-1][1] - hd[-2][1]) / 20.0, 1.0
                )

        if lank and rank:
            fs = abs(lank[0] - rank[0]) / (bw + 1e-6)
            feats["step_ready_stance"] = min(max(fs - 0.3, 0) * 2, 1.0)

        if person_id not in self.prev_keypoints:
            self.prev_keypoints[person_id] = keypoints
        pk = self.prev_keypoints[person_id]

        def wrist_v(curr, pidx):
            px, py, pc = pk[pidx]
            if curr is None or pc < 0.25:
                return 0.0
            return np.hypot(curr[0] - px, curr[1] - py) / (bh + 1e-6)

        feats["wrist_speed"] = min(max(wrist_v(lwrist, 9), wrist_v(rwrist, 10)) * 3.0, 1.0)
        self.prev_keypoints[person_id] = keypoints
        return feats

class AppearanceExtractor:
    """
    Batched + FP16 appearance extractor.
    """
    def __init__(self, device="cpu"):
        self.device = device
        self.use_half = USE_HALF and device != "cpu"
        base = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model = torch.nn.Sequential(*(list(base.children())[:-1])).to(device)
        if self.use_half:
            self.model = self.model.half()
        self.model.eval()
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self._emb_cache  = {}
        self._hist_cache = {}

    def _preprocess_crop(self, frame, bbox, pad_w=0.12, pad_h=0.04):
        x1, y1, x2, y2 = map(int, bbox)
        H, W = frame.shape[:2]
        bw, bh = x2 - x1, y2 - y1
        x1 = max(0, x1 + int(bw * pad_w))
        x2 = min(W, x2 - int(bw * pad_w))
        y1 = max(0, y1 + int(bh * pad_h))
        y2 = min(H, y2 - int(bh * pad_h))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            return None
        crop = cv2.resize(crop, (128, 256), interpolation=cv2.INTER_LINEAR)
        crop = crop[:, :, ::-1].astype(np.float32) / 255.0
        crop = (crop - self.mean) / self.std
        return crop.transpose(2, 0, 1)

    def extract_batch(self, frame, bboxes_and_pids, frame_idx):
        results = {}
        to_infer = []

        for pid, bbox in bboxes_and_pids:
            cached = self._emb_cache.get(pid)
            if cached and (frame_idx - cached[0]) < SKIP_EMBED_EVERY:
                results[pid] = cached[1]
                continue
            crop = self._preprocess_crop(frame, bbox)
            if crop is not None:
                to_infer.append((pid, crop))

        if to_infer:
            pids_batch  = [x[0] for x in to_infer]
            crops_batch = np.stack([x[1] for x in to_infer])
            t = torch.from_numpy(crops_batch).to(self.device)
            if self.use_half:
                t = t.half()
            with torch.no_grad():
                feats = self.model(t).squeeze(-1).squeeze(-1)
                if self.use_half:
                    feats = feats.float()
                feats = feats.cpu().numpy()
            for pid, feat in zip(pids_batch, feats):
                n = np.linalg.norm(feat)
                emb = feat / n if n > 0 else feat
                self._emb_cache[pid] = (frame_idx, emb)
                results[pid] = emb

        return results

    def extract(self, frame, bbox):
        """Single extract - legacy fallback"""
        crop = self._preprocess_crop(frame, bbox)
        if crop is None:
            return None
        t = torch.from_numpy(crop[None]).to(self.device)
        if self.use_half:
            t = t.half()
        with torch.no_grad():
            feat = self.model(t).squeeze().float().cpu().numpy()
        n = np.linalg.norm(feat)
        return feat / n if n > 0 else feat

    def color_hist(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        H, W = frame.shape[:2]
        bw, bh = x2 - x1, y2 - y1
        x1 = max(0, x1 + int(bw * 0.15))
        x2 = min(W, x2 - int(bw * 0.15))
        y1 = max(0, y1 + int(bh * 0.10))
        y2 = min(H, y2 - int(bh * 0.10))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [24, 24], [0, 180, 0, 256])
        hist = cv2.normalize(hist, hist).flatten().astype(np.float32)
        n = np.linalg.norm(hist)
        return hist / n if n > 0 else hist

class LostTrackMemory:
    """
    Memory bank
    """
    def __init__(self, frame_w, frame_h):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.memory = {}
        self.max_age_frames = 360
        self.reuse_distance_ratio = 0.22
        self.confirmed_thresh = 0.52
        self.normal_thresh = 0.74

    def update_lost(self, canonical_id, rec, frame_idx):
        if rec is None:
            return
        self.memory[canonical_id] = {
            "canonical_id": canonical_id,
            "embedding": None if rec.get("embedding") is None else rec["embedding"].copy(),
            "color_hist": None if rec.get("color_hist") is None else rec["color_hist"].copy(),
            "last_bbox": tuple(map(float, rec.get("last_bbox", (0, 0, 0, 0)))),
            "last_center": tuple(map(float, rec.get("last_center", (0, 0)))),
            "last_area": float(rec.get("last_area", 0.0)),
            "lost_frame": frame_idx,
            "confirmed": bool(rec.get("confirmed", False)),
        }

    def mark_visible(self, canonical_id):
        self.memory.pop(canonical_id, None)

    def _sim(self, a, b):
        if a is None or b is None:
            return 0.0
        return float(np.dot(a, b))

    def match(self, bbox, embedding, color_hist, frame_idx, used_canonicals, visible_now):
        center = bbox_center(bbox)
        area = bbox_area(bbox)
        best_id = None
        best_score = -1.0

        for cid, mem in list(self.memory.items()):
            if cid in used_canonicals or cid in visible_now:
                continue

            age = frame_idx - mem["lost_frame"]
            if age < 1 or age > self.max_age_frames:
                continue

            move_dist = np.hypot(center[0] - mem["last_center"][0], center[1] - mem["last_center"][1]) / (self.frame_w + 1e-6)
            pos_threshold = self.reuse_distance_ratio * 1.6 if mem["confirmed"] else self.reuse_distance_ratio
            if move_dist > pos_threshold:
                continue

            size_ratio = min(area, mem["last_area"] + 1e-6) / max(area, mem["last_area"] + 1e-6)
            if size_ratio < 0.35:
                continue

            app_sim = self._sim(embedding, mem["embedding"])
            hist_sim = self._sim(color_hist, mem["color_hist"])
            pos_score = max(0.0, 1.0 - move_dist / (self.reuse_distance_ratio + 1e-6))

            score = 0.58 * app_sim + 0.22 * hist_sim + 0.10 * size_ratio + 0.10 * pos_score
            thresh = self.confirmed_thresh if mem["confirmed"] else self.normal_thresh

            if mem["confirmed"]:
                if app_sim < 0.28 and hist_sim < 0.18:
                    continue
            else:
                if app_sim < 0.52:
                    continue
                if hist_sim < 0.28:
                    continue

            if score >= thresh and score > best_score:
                best_score = score
                best_id = cid

        return best_id, best_score

    def cleanup(self, frame_idx):
        stale = [cid for cid, mem in self.memory.items() if frame_idx - mem["lost_frame"] > self.max_age_frames]
        for cid in stale:
            self.memory.pop(cid, None)

class ReIDManager:
    def __init__(self, frame_w, frame_h):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.registry = {}
        self.alias_to_canonical = {}
        self.visible_now = set()
        self.prev_visible_now = set()
        self.last_seen_frame = {}
        self.min_missing = 2
        self.max_missing = 900
        self.max_move_ratio = 0.42
        self.confirmed_match_thresh = 0.54
        self.normal_match_thresh = 0.82
        self.memory = LostTrackMemory(frame_w, frame_h)

    def start_frame(self):
        self.prev_visible_now = set(self.visible_now)
        self.visible_now = set()

    def end_frame(self, current_frame, visible_ids):
        current_visible = set(visible_ids)
        disappeared = self.prev_visible_now - current_visible

        for pid in disappeared:
            canon = self._canon(pid)
            rec = self.registry.get(canon)
            self.memory.update_lost(canon, rec, current_frame)

        self.visible_now = current_visible
        for pid in current_visible:
            canon = self._canon(pid)
            self.last_seen_frame[canon] = current_frame
            rec = self.registry.get(canon)
            if rec is not None:
                self.memory.mark_visible(canon)

    def _canon(self, pid):
        return self.alias_to_canonical.get(pid, pid)

    def register(self, pid, embedding, color_hist, bbox, frame_idx,
                 confirmed=False, peak_risk=0.0, confirm_reasons=None):
        pid = self._canon(pid)
        center = bbox_center(bbox)
        area = bbox_area(bbox)
        if pid not in self.registry:
            self.registry[pid] = {
                "embedding": embedding.copy() if embedding is not None else None,
                "color_hist": color_hist.copy() if color_hist is not None else None,
                "last_frame": frame_idx,
                "last_bbox": tuple(map(float, bbox)),
                "last_center": center,
                "last_area": area,
                "confirmed": confirmed,
                "peak_risk": peak_risk,
                "confirm_reasons": list(confirm_reasons) if confirm_reasons else [],
            }
            self.alias_to_canonical[pid] = pid
            self.last_seen_frame[pid] = frame_idx
            self.memory.mark_visible(pid)
            return

        rec = self.registry[pid]
        if embedding is not None:
            if rec["embedding"] is None:
                rec["embedding"] = embedding.copy()
            else:
                merged = 0.85 * rec["embedding"] + 0.15 * embedding
                n = np.linalg.norm(merged)
                rec["embedding"] = merged / n if n > 0 else merged

        if color_hist is not None:
            if rec["color_hist"] is None:
                rec["color_hist"] = color_hist.copy()
            else:
                merged_hist = 0.80 * rec["color_hist"] + 0.20 * color_hist
                n = np.linalg.norm(merged_hist)
                rec["color_hist"] = merged_hist / n if n > 0 else merged_hist

        rec["last_frame"] = frame_idx
        rec["last_bbox"] = tuple(map(float, bbox))
        rec["last_center"] = center
        rec["last_area"] = area
        if confirmed:
            rec["confirmed"] = True
            rec["peak_risk"] = max(rec["peak_risk"], peak_risk)
            if confirm_reasons:
                rec["confirm_reasons"] = list(confirm_reasons)
        self.last_seen_frame[pid] = frame_idx
        self.memory.mark_visible(pid)

    def _appearance_similarity(self, emb_a, emb_b):
        if emb_a is None or emb_b is None:
            return 0.0
        return float(np.dot(emb_a, emb_b))

    def _hist_similarity(self, hist_a, hist_b):
        if hist_a is None or hist_b is None:
            return 0.0
        return float(np.dot(hist_a, hist_b))

    def resolve(self, tracker_id, embedding, color_hist, bbox, frame_idx, used_canonicals):
        tracker_id = self._canon(tracker_id)
        if tracker_id in self.registry:
            canon = tracker_id
            self.memory.mark_visible(canon)
            return canon, False, self.registry[canon]

        center = bbox_center(bbox)
        area = bbox_area(bbox)

        best_id = None
        best_score = -1.0

        for cid, rec in self.registry.items():
            if cid in used_canonicals:
                continue

            age = frame_idx - rec["last_frame"]
            if age < self.min_missing or age > self.max_missing:
                continue

            if cid in self.visible_now or cid in self.prev_visible_now:
                continue

            prev_center = rec["last_center"]
            move_dist = np.hypot(center[0] - prev_center[0], center[1] - prev_center[1]) / (self.frame_w + 1e-6)
            if move_dist > self.max_move_ratio:
                continue

            size_ratio = min(area, rec["last_area"] + 1e-6) / max(area, rec["last_area"] + 1e-6)
            if size_ratio < 0.35:
                continue

            app_sim = self._appearance_similarity(embedding, rec["embedding"])
            hist_sim = self._hist_similarity(color_hist, rec["color_hist"])
            pos_score = max(0.0, 1.0 - move_dist / (self.max_move_ratio + 1e-6))

            score = 0.60 * app_sim + 0.22 * hist_sim + 0.10 * size_ratio + 0.08 * pos_score
            thresh = self.confirmed_match_thresh if rec["confirmed"] else self.normal_match_thresh

            if rec["confirmed"]:
                if app_sim < 0.28 and hist_sim < 0.18:
                    continue
            else:
                if app_sim < 0.55 or hist_sim < 0.30:
                    continue

            if score >= thresh and score > best_score:
                best_score = score
                best_id = cid

        mem_id = None
        mem_score = -1.0
        if best_id is None:
            mem_id, mem_score = self.memory.match(
                bbox=bbox,
                embedding=embedding,
                color_hist=color_hist,
                frame_idx=frame_idx,
                used_canonicals=used_canonicals,
                visible_now=self.visible_now | self.prev_visible_now,
            )
            if mem_id is not None:
                best_id = mem_id
                best_score = mem_score

        if best_id is None:
            return tracker_id, False, None

        self.alias_to_canonical[tracker_id] = best_id
        self.memory.mark_visible(best_id)
        return best_id, True, self.registry.get(best_id)

    def is_confirmed(self, pid):
        pid = self._canon(pid)
        return self.registry.get(pid, {}).get("confirmed", False)

    def cleanup(self, current_frame):
        stale = [pid for pid, rec in self.registry.items() if current_frame - rec["last_frame"] > self.max_missing]
        for pid in stale:
            self.registry.pop(pid, None)
            self.alias_to_canonical.pop(pid, None)
            self.last_seen_frame.pop(pid, None)
        self.memory.cleanup(current_frame)

class ObjectTracker:
    def __init__(self, frame_w, frame_h):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.objects: dict = {}
        self._next_id = 0
        self.person_near_obj: dict = defaultdict(dict)
        self.person_vanished_objs: dict = defaultdict(set)

    def _match_objects(self, detections: list, frame_idx: int):
        matched_ids = set()
        used_dets   = set()

        for obj_id, rec in self.objects.items():
            if rec.get("missing_since") is not None:
                continue
            best_iou, best_det = 0.0, -1
            for di, (cls_id, x1, y1, x2, y2) in enumerate(detections):
                if di in used_dets:
                    continue
                if cls_id != rec["cls"]:
                    continue
                iou = bbox_iou(rec["bbox"], (x1, y1, x2, y2))
                if iou > best_iou:
                    best_iou = iou
                    best_det = di
            if best_iou > 0.25 and best_det >= 0:
                di = best_det
                cls_id, x1, y1, x2, y2 = detections[di]
                rec["bbox"]       = (x1, y1, x2, y2)
                rec["center"]     = bbox_center((x1, y1, x2, y2))
                rec["last_frame"] = frame_idx
                matched_ids.add(obj_id)
                used_dets.add(di)

        for di, (cls_id, x1, y1, x2, y2) in enumerate(detections):
            if di in used_dets:
                continue
            oid = self._next_id
            self._next_id += 1
            self.objects[oid] = {
                "cls":          cls_id,
                "bbox":         (x1, y1, x2, y2),
                "center":       bbox_center((x1, y1, x2, y2)),
                "first_frame":  frame_idx,
                "last_frame":   frame_idx,
                "missing_since": None,
            }

        for obj_id, rec in self.objects.items():
            if obj_id in matched_ids:
                rec["missing_since"] = None
                continue
            if rec.get("missing_since") is None and rec["last_frame"] < frame_idx:
                rec["missing_since"] = frame_idx

    def update(self, detections: list, all_kps: dict, frame_idx: int):
        self._match_objects(detections, frame_idx)

        diag = np.hypot(self.frame_w, self.frame_h)
        touch_px = OBJECT_TOUCH_RADIUS * self.frame_w

        for pid, kps in all_kps.items():
            wrists = []
            for idx in (9, 10):
                x, y, c = kps[idx]
                if c > 0.25:
                    wrists.append((float(x), float(y)))

            for obj_id, rec in self.objects.items():
                if rec.get("missing_since") is not None:
                    if obj_id in self.person_near_obj.get(pid, {}):
                        frames_ago = frame_idx - self.person_near_obj[pid][obj_id]
                        if frames_ago < OBJECT_MISSING_PATIENCE:
                            self.person_vanished_objs[pid].add(obj_id)
                    continue

                cx, cy = rec["center"]
                for wx, wy in wrists:
                    dist = np.hypot(wx - cx, wy - cy)
                    if dist < touch_px:
                        self.person_near_obj[pid][obj_id] = frame_idx
                        break

    def get_signals(self, pid: int, kps: np.ndarray, frame_idx: int) -> dict:
        near_obj = bool(self.person_near_obj.get(pid))
        vanished = len(self.person_vanished_objs.get(pid, set())) > 0

        grab_score = 0.0
        if self.objects:
            diag = np.hypot(self.frame_w, self.frame_h)
            touch_px = OBJECT_TOUCH_RADIUS * self.frame_w * 2.0
            for idx in (9, 10):
                x, y, c = kps[idx]
                if c < 0.2:
                    continue
                for rec in self.objects.values():
                    if rec.get("missing_since") is not None:
                        continue
                    cx, cy = rec["center"]
                    dist = np.hypot(x - cx, y - cy)
                    if dist < touch_px:
                        prox = max(0.0, 1.0 - dist / touch_px)
                        grab_score = max(grab_score, prox * float(c))

        return {
            "hand_near_object":  near_obj,
            "object_grab_score": float(grab_score),
            "object_vanished":   vanished,
        }

    def cleanup(self, frame_idx: int, max_missing_age: int = 180):
        stale = [oid for oid, rec in self.objects.items()
                 if rec.get("missing_since") is not None
                 and frame_idx - rec["missing_since"] > max_missing_age]
        for oid in stale:
            self.objects.pop(oid, None)

        for pid in list(self.person_vanished_objs.keys()):
            self.person_vanished_objs[pid] = {
                oid for oid in self.person_vanished_objs[pid]
                if oid in self.objects
                and frame_idx - self.objects[oid].get("missing_since", frame_idx) < max_missing_age
            }


class TheftDetectionPipeline:
    def __init__(self, video_path):
        self.video_path = video_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Device: {self.device}")

        self.pose_model = YOLO(POSE_MODEL_NAME)
        self.pose_model.conf = 0.40
        dummy = np.zeros((INFER_SIZE, INFER_SIZE, 3), dtype=np.uint8)
        self.pose_model(dummy, imgsz=INFER_SIZE, half=USE_HALF, verbose=False)
        print(f"[INFO] {POSE_MODEL_NAME} loaded + warmed up  (half={USE_HALF})")

        self.seg_model = YOLO(SEG_MODEL_NAME)
        self.seg_model.conf = 0.40
        self.seg_model(dummy, imgsz=INFER_SIZE, half=USE_HALF, verbose=False)
        print(f"[INFO] {SEG_MODEL_NAME} loaded + warmed up")

        self.use_sahi_obj = USE_SAHI_OBJECT_DETECTION and SAHI_AVAILABLE
        self.obj_model = None
        self.sahi_obj_model = None
        if self.use_sahi_obj:
            self.sahi_obj_model = AutoDetectionModel.from_pretrained(
                model_type="ultralytics",
                model_path=OBJ_MODEL_NAME,
                confidence_threshold=OBJ_CONF,
                device="cuda:0" if self.device == "cuda" else "cpu",
            )
            print(
                f"[INFO] {OBJ_MODEL_NAME} loaded with SAHI sliced inference "
                f"({SAHI_SLICE_WIDTH}x{SAHI_SLICE_HEIGHT}, overlap={SAHI_OVERLAP_RATIO})"
            )
        else:
            if USE_SAHI_OBJECT_DETECTION and not SAHI_AVAILABLE:
                print("[WARN] SAHI is not installed. Falling back to normal YOLO object detection. Install with: pip install sahi")
            self.obj_model = YOLO(OBJ_MODEL_NAME)
            self.obj_model.conf = OBJ_CONF
            self.obj_model(dummy, imgsz=INFER_SIZE, half=USE_HALF, verbose=False)
            print(f"[INFO] {OBJ_MODEL_NAME} loaded + warmed up")

        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.35,
            lost_track_buffer=90,
            minimum_matching_threshold=0.80,
            frame_rate=30,
        )
        print("[INFO] ByteTrack (extended buffer) loaded.")

        self.seg_masks      = {}
        self.last_seg_frame = -SEG_EVERY_N

        self.object_tracker: ObjectTracker | None = None
        self.last_obj_frame  = -OBJ_INFER_EVERY_N
        self._last_obj_dets: list = []

        self._fps_t0    = None
        self._fps_count = 0
        self._fps_val   = 0.0
        self.classifier = TheftDetectionLSTM(input_dim=17, hidden_dim=128).to(self.device)
        self.classifier.eval()

        self.appearance_extractor = AppearanceExtractor(self.device)
        self.reid = None
        self.skeleton_analyzer = SkeletonAnalyzer()

        self.seq_length = 20
        self.seq_buffer = defaultdict(lambda: deque(maxlen=self.seq_length))
        self.person_history = defaultdict(lambda: deque(maxlen=60))
        self.person_speed_h = defaultdict(lambda: deque(maxlen=10))
        self.speed_window = defaultdict(lambda: deque(maxlen=30))
        self.prev_bbox = {}

        self.grab_history = defaultdict(lambda: deque(maxlen=8))
        self.wrist_speed_history = defaultdict(lambda: deque(maxlen=8))
        self.pull_history = defaultdict(lambda: deque(maxlen=8))
        self.table_history = defaultdict(lambda: deque(maxlen=8))
        self.lean_history = defaultdict(lambda: deque(maxlen=8))
        self.confirm_buffer = defaultdict(lambda: deque(maxlen=8))
        self.cluster_dominance_buffer = defaultdict(lambda: deque(maxlen=8))
        self.near_confirm_cooldown = defaultdict(int)

        self.confirmed_thieves = {}
        self.active_alarms = {}
        self.risk_scores = {}

        self.frame_idx = 0
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def _transfer_state(self, old_id, canonical_id):
        if old_id == canonical_id:
            return

        for store in [
            self.seq_buffer, self.person_history, self.person_speed_h, self.speed_window,
            self.grab_history, self.wrist_speed_history, self.pull_history,
            self.table_history, self.lean_history, self.confirm_buffer
        ]:
            if old_id in store:
                if canonical_id in store and isinstance(store[old_id], deque):
                    old_vals = list(store[old_id])
                    for item in old_vals:
                        store[canonical_id].append(item)
                elif canonical_id not in store:
                    store[canonical_id] = store[old_id]
                try:
                    del store[old_id]
                except KeyError:
                    pass

        if old_id in self.prev_bbox:
            self.prev_bbox[canonical_id] = self.prev_bbox.pop(old_id)

        if old_id in self.confirmed_thieves and canonical_id not in self.confirmed_thieves:
            self.confirmed_thieves[canonical_id] = self.confirmed_thieves.pop(old_id)

        if old_id in self.active_alarms and canonical_id not in self.active_alarms:
            self.active_alarms[canonical_id] = self.active_alarms.pop(old_id)

        self.skeleton_analyzer.transfer_state(old_id, canonical_id)

    def extract_features(self, person_id, bbox, keypoints, all_persons, confirmed_set, frame_h, frame_w):
        cx, cy = bbox_center(bbox)
        area = bbox_area(bbox)
        skel = self.skeleton_analyzer.extract(person_id, keypoints, bbox, frame_h, frame_w)

        speed = 0.0
        if person_id in self.prev_bbox:
            pcx, pcy = bbox_center(self.prev_bbox[person_id])
            speed = np.hypot(cx - pcx, cy - pcy) / (frame_h + 1e-6)
        self.person_speed_h[person_id].append(speed)

        self.person_history[person_id].append((cx, cy))
        loit = loitering_score(self.person_history[person_id])

        social_others = {
            pid: bb for pid, bb in all_persons.items()
            if pid != person_id and pid not in confirmed_set
        }

        social_dist, approach_rate = 1.0, 0.0
        if social_others:
            my_area = bbox_area(bbox)
            valid = {
                pid: bb for pid, bb in social_others.items()
                if 0.25 < bbox_area(bb) / (my_area + 1e-6) < 4.0
            }
            if valid:
                ocs = np.array([bbox_center(bb) for bb in valid.values()])
                mc = np.array([[cx, cy]])
                dists = cdist(mc, ocs)[0]
                md = dists.min() / (frame_w + 1e-6)
                social_dist = min(md, 1.0)
                cpid = list(valid.keys())[dists.argmin()]
                if cpid in self.prev_bbox:
                    px2, py2 = bbox_center(self.prev_bbox[cpid])
                    pd = np.hypot(cx - px2, cy - py2)
                    approach_rate = max(0.0, min((pd - md * frame_w) / (frame_h + 1e-6), 1.0))

        occlusion = 0.0
        x1, y1, x2, y2 = bbox
        for _, bb2 in social_others.items():
            ox1, oy1, ox2, oy2 = bb2
            ix = max(0, min(x2, ox2) - max(x1, ox1))
            iy = max(0, min(y2, oy2) - max(y1, oy1))
            inter = ix * iy
            union = area + bbox_area(bb2) - inter + 1e-6
            occlusion = max(occlusion, inter / union)

        sudden_stop = 0.0
        if len(self.person_speed_h[person_id]) >= 3:
            hs = list(self.person_speed_h[person_id])
            if hs[-2] > 0.03 and hs[-1] < 0.005:
                sudden_stop = 1.0

        return [
            skel["wrist_to_pocket_dist"],
            skel["elbow_angle_left"],
            skel["elbow_angle_right"],
            skel["arm_extension"],
            skel["head_turn_speed"],
            skel["gaze_direction_change"],
            skel["spine_lean_forward"],
            skel["wrist_above_table"],
            skel["wrist_moving_toward_body"],
            skel["grab_motion_score"],
            speed,
            loit,
            social_dist,
            approach_rate,
            occlusion,
            sudden_stop,
            skel["wrist_speed"],
        ]

    def compute_rule_based_risk(self, person_id, features):
        risk, reason = 0.0, []

        wrist_pocket = features[0]
        arm_ext = features[3]
        head_turn = features[4]
        spine_lean = features[6]
        wrist_table = features[7]
        wrist_pull = features[8]
        grab_score = features[9]
        speed = features[10]
        loit = features[11]
        social_dist = features[12]
        approach = features[13]
        occlusion = features[14]
        sudden_stop = features[15]
        wrist_speed = features[16]

        self.grab_history[person_id].append(grab_score)
        self.wrist_speed_history[person_id].append(wrist_speed)
        self.pull_history[person_id].append(wrist_pull)
        self.table_history[person_id].append(wrist_table)
        self.lean_history[person_id].append(spine_lean)

        grab_hist = list(self.grab_history[person_id])
        ws_hist = list(self.wrist_speed_history[person_id])
        pull_hist = list(self.pull_history[person_id])
        table_hist = list(self.table_history[person_id])
        lean_hist = list(self.lean_history[person_id])

        mean_grab = float(np.mean(grab_hist)) if grab_hist else 0.0
        max_ws = float(np.max(ws_hist)) if ws_hist else 0.0
        mean_pull = float(np.mean(pull_hist)) if pull_hist else 0.0
        mean_table = float(np.mean(table_hist)) if table_hist else 0.0
        mean_lean = float(np.mean(lean_hist)) if lean_hist else 0.0

        suspicion = 0.0

        if spine_lean > 0.24 and wrist_table > 0.24 and arm_ext > 0.24:
            suspicion += 0.32; reason.append("DirectGrabPose")
        if grab_score > 0.42:
            suspicion += 0.22; reason.append(f"GrabPosture:{grab_score:.2f}")
        elif grab_score > 0.30:
            suspicion += 0.12; reason.append(f"WeakGrab:{grab_score:.2f}")
        if wrist_pull > 0.26 and arm_ext > 0.24:
            suspicion += 0.18; reason.append("WristRetract")
        if wrist_pocket > 0.58 and head_turn > 0.25:
            suspicion += 0.16; reason.append("PocketReach+LookAround")
        if loit > 0.55 and speed < 0.004:
            suspicion += 0.10; reason.append("Loitering")
        if arm_ext > 0.42 and social_dist < 0.10:
            suspicion += 0.10; reason.append("ArmExtend+VeryClose")
        if occlusion > 0.22 and approach > 0.012:
            suspicion += 0.10; reason.append("Blocking+Approaching")
        if sudden_stop and grab_score > 0.18:
            suspicion += 0.10; reason.append("SuddenStop+Grab")

        motion_now = (
            (wrist_speed > 0.42 and grab_score > 0.20) or
            (wrist_speed > 0.38 and wrist_pull > 0.20) or
            (wrist_speed > 0.38 and wrist_table > 0.20) or
            (wrist_pull > 0.24 and wrist_speed > 0.22 and grab_score > 0.15) or
            (speed > 0.016 and wrist_speed > 0.20 and grab_score > 0.26)
        )
        sustained_motion = (
            max_ws > 0.30 and mean_grab > 0.28 and
            (mean_pull > 0.18 or mean_table > 0.22 or mean_lean > 0.20)
        )
        commit_pose = (
            mean_grab > 0.34 and mean_table > 0.24 and
            (arm_ext > 0.24 or mean_lean > 0.22)
        )

        pose_only_gate = (
            spine_lean  > 0.22 and
            wrist_table > 0.22 and
            arm_ext     > 0.22 and
            grab_score  > 0.28
        )

        sustained_pose_gate = (
            mean_lean  > 0.20 and
            mean_table > 0.20 and
            mean_grab  > 0.25 and
            len(grab_hist) >= 4
        )

        motion_gate = motion_now or (sustained_motion and commit_pose) or pose_only_gate or sustained_pose_gate

        if motion_gate:
            reason.append("MotionGate:YES")
            risk += min(suspicion, 0.58)
            if wrist_speed > 0.42 and grab_score > 0.24:
                risk += 0.22; reason.append(f"FastGrab:ws={wrist_speed:.2f}")
            if wrist_speed > 0.32 and wrist_pull > 0.22:
                risk += 0.20; reason.append("FastPull")
            if wrist_speed > 0.36 and wrist_table > 0.22:
                risk += 0.18; reason.append("FastTableGrab")
            if speed > 0.018 and grab_score > 0.26:
                risk += min(speed * 5.0 * max(grab_score, 0.25), 0.24)
                reason.append(f"SpeedGrab:{speed:.3f}x{grab_score:.2f}")
            if sustained_motion:
                risk += 0.12; reason.append("SustainedHandMotion")
            if commit_pose:
                risk += 0.10; reason.append("CommittedGrabPose")
            if speed > 0.035 and loit > 0.30:
                risk += 0.16; reason.append("FleeSprint")

            if pose_only_gate:
                pose_strength = (spine_lean + wrist_table + arm_ext + grab_score) / 4.0
                risk += min(pose_strength * 0.60, 0.45)
                reason.append(f"SlowDeliberateGrab:{pose_strength:.2f}")
            if sustained_pose_gate:
                risk += 0.15; reason.append("SustainedGrabPose")
        else:
            risk += min(suspicion * 0.35, 0.22)
            reason.append("MotionGate:NO")

        temporal_commit = (
            mean_grab > 0.34 and max_ws > 0.28 and
            (mean_pull > 0.20 or wrist_pull > 0.24)
        )
        if temporal_commit:
            risk += 0.10
            reason.append("TemporalCommit")

        return min(risk, 1.0), reason

    def suppress_nearby_false_positives(self, frame_risks, person_debug, frame_w):
        pids = list(frame_risks.keys())
        if not pids:
            return

        adjacency = {pid: set() for pid in pids}
        depth_occluded = {}   

        for i in range(len(pids)):
            for j in range(i + 1, len(pids)):
                a, b = pids[i], pids[j]
                ba = person_debug[a]["bbox"]
                bb = person_debug[b]["bbox"]
                iou = bbox_iou(ba, bb)
                ac = bbox_center(ba)
                bc = bbox_center(bb)
                dist = np.hypot(ac[0] - bc[0], ac[1] - bc[1]) / (frame_w + 1e-6)
                if iou > 0.02 or dist < 0.16:
                    adjacency[a].add(b)
                    adjacency[b].add(a)

                area_a = bbox_area(ba)
                area_b = bbox_area(bb)
                ax1,ay1,ax2,ay2 = ba
                bx1,by1,bx2,by2 = bb
                ix = max(0, min(ax2,bx2) - max(ax1,bx1))
                iy = max(0, min(ay2,by2) - max(ay1,by1))
                inter = ix * iy
                if inter > 0:
                    min_area = min(area_a, area_b) + 1e-6
                    overlap_ratio = inter / min_area
                    if overlap_ratio > 0.45:
                        behind = b if area_b < area_a else a
                        front  = a if area_b < area_a else b
                        depth_occluded.setdefault(front, set()).add(behind)
                        depth_occluded.setdefault(behind, set())  

        visited = set()
        for pid in pids:
            if pid in visited:
                continue
            stack = [pid]
            cluster = []
            while stack:
                cur = stack.pop()
                if cur in visited:
                    continue
                visited.add(cur)
                cluster.append(cur)
                stack.extend(list(adjacency[cur] - visited))

            if len(cluster) <= 1:
                person_debug[cluster[0]]["dominant_actor"] = True
                continue

            def dominance_score(x):
                feats = person_debug[x]["feats"]
                motion_bonus = 0.28 if person_debug[x]["motion_evidence"] else 0.0
                confirm_bonus = 0.18 if x in self.confirmed_thieves else 0.0
                return (
                    0.42 * person_debug[x]["interaction_score"] +
                    0.18 * feats[16] +
                    0.16 * feats[8] +
                    0.12 * feats[7] +
                    0.12 * feats[9] +
                    motion_bonus +
                    confirm_bonus
                )

            ranked = sorted(cluster, key=dominance_score, reverse=True)
            dominant = ranked[0]
            dom_score = dominance_score(dominant)
            person_debug[dominant]["dominant_actor"] = True

            for other in ranked[1:]:
                person_debug[other]["dominant_actor"] = False
                other_score = dominance_score(other)
                feats = person_debug[other]["feats"]
                strong_independent_motion = (
                    feats[16] > 0.60 and feats[8] > 0.35 and feats[9] > 0.42
                    and feats[7] > 0.25
                )
                gap = dom_score - other_score

                is_depth_behind_dominant = other in depth_occluded.get(dominant, set())
                motion_ev_other = person_debug[other].get("motion_evidence", False)

                if is_depth_behind_dominant and not motion_ev_other:
                    risk_o, reasons_o = frame_risks[other]
                    capped = 0.08 if other not in self.confirmed_thieves else 0.22
                    frame_risks[other] = (min(risk_o, capped),
                                          reasons_o + ["Suppressed:DepthOccludedBystander"])
                    person_debug[other]["suppressed"] = True
                    self.near_confirm_cooldown[other] = max(
                        self.near_confirm_cooldown.get(other, 0), 30)
                    continue

                if gap >= 0.06 and not strong_independent_motion:
                    risk_o, reasons_o = frame_risks[other]
                    if gap >= 0.20:
                        capped = 0.12 if other not in self.confirmed_thieves else 0.28
                        cooldown = 20
                    else:
                        capped = 0.18 if other not in self.confirmed_thieves else 0.32
                        cooldown = 12
                    frame_risks[other] = (min(risk_o, capped), reasons_o + ["Suppressed:ClusterNonDominant"])
                    person_debug[other]["suppressed"] = True
                    self.near_confirm_cooldown[other] = max(self.near_confirm_cooldown.get(other, 0), cooldown)

            if not person_debug[dominant]["motion_evidence"]:
                risk_d, reasons_d = frame_risks[dominant]
                frame_risks[dominant] = (min(risk_d, 0.42), reasons_d + ["ClusterDominantButNoMotion"])

    def draw_person(self, frame, person_id, bbox, keypoints, risk_score, reasons, seg_mask=None):
        x1, y1, x2, y2 = map(int, bbox)
        confirmed = person_id in self.confirmed_thieves

        if seg_mask is not None:
            overlay = frame.copy()
            if confirmed:
                mask_color = (0, 0, 220)
                alpha = 0.45
            elif risk_score > 0.5:
                mask_color = (0, 80, 220)
                alpha = 0.30
            else:
                mask_color = (0, 180, 0)
                alpha = 0.18
            colored = np.zeros_like(frame, dtype=np.uint8)
            colored[seg_mask == 1] = mask_color
            cv2.addWeighted(colored, alpha, overlay, 1 - alpha, 0, overlay)
            bslice = np.s_[max(0, y1):min(frame.shape[0], y2+5),
                           max(0, x1):min(frame.shape[1], x2+5)]
            frame[bslice] = overlay[bslice]

        if confirmed:
            sc, dc, th = (0, 0, 255), (0, 0, 255), 3
        elif risk_score > 0.5:
            sc, dc, th = (0, 100, 255), (0, 50, 255), 2
        else:
            sc, dc, th = (0, 255, 255), (0, 255, 0), 2

        for i, j in SKELETON_CONNECTIONS:
            xi, yi, ci = keypoints[i]
            xj, yj, cj = keypoints[j]
            if ci > 0.3 and cj > 0.3:
                cv2.line(frame, (int(xi), int(yi)), (int(xj), int(yj)), sc, th)
        for idx in range(17):
            xk, yk, ck = keypoints[idx]
            if ck > 0.3:
                cv2.circle(frame, (int(xk), int(yk)), 5 if confirmed else 4, dc, -1)

        if confirmed:
            cv2.rectangle(frame, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (0, 0, 100), 4)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.rectangle(frame, (x1, y1 - 22), (x2, y1), (0, 0, 200), -1)
            cv2.putText(frame, f"CONFIRMED THIEF ID:{person_id}", (x1 + 4, y1 - 6), self.font, 0.45, (255, 255, 255), 1)
            bw = x2 - x1
            bf = int(bw * risk_score)
            cv2.rectangle(frame, (x1, y2 + 3), (x2, y2 + 10), (40, 40, 40), -1)
            cv2.rectangle(frame, (x1, y2 + 3), (x1 + bf, y2 + 10), (0, 0, 255), -1)
            for li, r in enumerate(reasons[:3]):
                cv2.putText(frame, r, (x1, y1 - 28 - li * 16), self.font, 0.4, (0, 100, 255), 1)
        elif risk_score > 0.6:
            inten = int(100 + 155 * risk_score)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, inten), 3)
            for li, r in enumerate(reasons[:3]):
                cv2.putText(frame, r, (x1, y1 - 12 - li * 18), self.font, 0.45, (0, 0, 255), 1)
            bw = x2 - x1
            bf = int(bw * risk_score)
            cv2.rectangle(frame, (x1, y2 + 3), (x2, y2 + 10), (50, 50, 50), -1)
            cv2.rectangle(frame, (x1, y2 + 3), (x1 + bf, y2 + 10), (0, 0, 255), -1)
            cv2.putText(frame, f"ID:{person_id} Risk:{risk_score:.2f}", (x1, y1 - 5), self.font, 0.5, (0, 0, 255), 1)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.putText(frame, f"ID:{person_id} Risk:{risk_score:.2f}", (x1, y1 - 5), self.font, 0.5, (0, 255, 255), 1)

        pts = list(self.person_history[person_id])
        for i in range(1, len(pts)):
            t = i / len(pts)
            c = (0, 0, int(180 + 75 * t)) if confirmed else (int(255 * (1 - t)), int(200 * t), int(255 * risk_score))
            cv2.line(frame, (int(pts[i - 1][0]), int(pts[i - 1][1])), (int(pts[i][0]), int(pts[i][1])), c, 2 if confirmed else 1)

    def _is_theft_target(self, cls_id, cls_name=None):
        if cls_id in THEFT_TARGET_CLASSES:
            return True
        if cls_name is not None and str(cls_name).lower() in TARGET_CLASS_NAMES:
            return True
        return False

    def _detect_theft_targets(self, frame):
        detections = []

        if self.use_sahi_obj and self.sahi_obj_model is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = get_sliced_prediction(
                rgb,
                self.sahi_obj_model,
                slice_height=SAHI_SLICE_HEIGHT,
                slice_width=SAHI_SLICE_WIDTH,
                overlap_height_ratio=SAHI_OVERLAP_RATIO,
                overlap_width_ratio=SAHI_OVERLAP_RATIO,
                perform_standard_pred=SAHI_PERFORM_STANDARD_PRED,
                postprocess_type="NMS",
                postprocess_match_metric="IOS",
                postprocess_match_threshold=0.50,
                # batch_size=SAHI_BATCH_SIZE,  <-- REMOVED to fix the TypeError
            )

            for pred in result.object_prediction_list:
                cls_name = getattr(pred.category, "name", None)
                try:
                    cls_id = int(pred.category.id)
                except (TypeError, ValueError):
                    cls_id = next((k for k, v in COCO_NAMES.items() if v == str(cls_name).lower()), -1)

                if not self._is_theft_target(cls_id, cls_name):
                    continue

                x1, y1, x2, y2 = pred.bbox.to_xyxy()
                detections.append((cls_id, float(x1), float(y1), float(x2), float(y2)))

            return detections

        obj_results = self.obj_model(
            frame,
            imgsz=INFER_SIZE,
            half=USE_HALF,
            verbose=False,
            classes=list(THEFT_TARGET_CLASSES),
        )[0]
        if obj_results.boxes is not None:
            for obox in obj_results.boxes:
                cls_id = int(obox.cls[0])
                if cls_id not in THEFT_TARGET_CLASSES:
                    continue
                ox1, oy1, ox2, oy2 = obox.xyxy[0].cpu().numpy()
                detections.append((cls_id, float(ox1), float(oy1), float(ox2), float(oy2)))
        return detections

    def draw_objects(self, frame, detections):
        for cls_id, ox1, oy1, ox2, oy2 in detections:
            label = COCO_NAMES.get(cls_id, f"obj{cls_id}")
            cv2.rectangle(frame, (int(ox1), int(oy1)), (int(ox2), int(oy2)), (0, 220, 255), 2)
            cv2.putText(
                frame,
                label,
                (int(ox1), max(12, int(oy1) - 5)),
                self.font,
                0.45,
                (0, 220, 255),
                1,
            )

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open: {self.video_path}")
            return

        import threading, queue as _queue, time as _time

        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[INFO] Video: {fw}x{fh}")
        target_h = 720
        target_w = int(fw * target_h / fh)
        self.reid = ReIDManager(target_w, target_h)
        self.object_tracker = ObjectTracker(target_w, target_h)

        _frame_q = _queue.Queue(maxsize=4)
        def _reader():
            while True:
                ret, f = cap.read()
                _frame_q.put((ret, f))
                if not ret:
                    break
        _t = threading.Thread(target=_reader, daemon=True)
        _t.start()

        import time as _time_mod
        _fps_timer = _time_mod.perf_counter()
        _fps_frames = 0

        while True:
            ret, frame = _frame_q.get()
            if not ret:
                break

            self.frame_idx += 1
            frame = cv2.resize(frame, (target_w, target_h))
            h, w = frame.shape[:2]
            self.reid.start_frame()

            _fps_frames += 1
            if _fps_frames >= 30:
                _now = _time_mod.perf_counter()
                self._fps_val = _fps_frames / (_now - _fps_timer + 1e-6)
                _fps_timer  = _now
                _fps_frames = 0

            for pid in list(self.near_confirm_cooldown.keys()):
                self.near_confirm_cooldown[pid] = max(0, self.near_confirm_cooldown[pid] - 1)
                if self.near_confirm_cooldown[pid] == 0:
                    del self.near_confirm_cooldown[pid]

            results = self.pose_model(
                frame, imgsz=INFER_SIZE, half=USE_HALF, verbose=False
            )[0]
            persons_raw = {}
            boxes_sv, confs_sv, cls_sv = [], [], []

            if results.keypoints is not None and len(results.boxes) > 0:
                for idx, (box, kps) in enumerate(zip(results.boxes, results.keypoints)):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    kps_np = kps.data[0].cpu().numpy()
                    persons_raw[idx] = ((x1, y1, x2, y2), kps_np)
                    boxes_sv.append([x1, y1, x2, y2])
                    confs_sv.append(conf)
                    cls_sv.append(0)

            run_seg = (self.frame_idx - self.last_seg_frame) >= SEG_EVERY_N
            if run_seg:
                self.last_seg_frame = self.frame_idx
                seg_mask_map = {}
                seg_results = self.seg_model(
                    frame, imgsz=INFER_SIZE, half=USE_HALF,
                    verbose=False, classes=[0]
                )[0]
                if seg_results.masks is not None and len(seg_results.boxes) > 0:
                    seg_masks_data = seg_results.masks.data.cpu().numpy()
                    for si, sbox in enumerate(seg_results.boxes):
                        sx1, sy1, sx2, sy2 = sbox.xyxy[0].cpu().numpy()
                        best_pi, best_iou = -1, 0.0
                        for pi, (pb, _) in persons_raw.items():
                            iou_val = bbox_iou((sx1, sy1, sx2, sy2), pb)
                            if iou_val > best_iou:
                                best_iou, best_pi = iou_val, pi
                        if best_pi >= 0 and best_iou > 0.35:
                            raw_mask = seg_masks_data[si]
                            full_mask = cv2.resize(
                                raw_mask, (w, h), interpolation=cv2.INTER_NEAREST
                            )
                            seg_mask_map[best_pi] = (full_mask > 0.5).astype(np.uint8)
                self.seg_masks[self.frame_idx] = seg_mask_map
                if len(self.seg_masks) > 5:
                    oldest = min(self.seg_masks)
                    del self.seg_masks[oldest]
            else:
                last_f = max(self.seg_masks) if self.seg_masks else -1
                seg_mask_map = self.seg_masks.get(last_f, {})
                self.seg_masks[self.frame_idx] = seg_mask_map

            run_obj = (self.frame_idx - self.last_obj_frame) >= OBJ_INFER_EVERY_N
            if run_obj:
                self.last_obj_frame = self.frame_idx
                self._last_obj_dets = self._detect_theft_targets(frame)

            if not boxes_sv:
                self.reid.end_frame(self.frame_idx, [])
                cv2.putText(frame, f"Frame: {self.frame_idx}", (10, 30), self.font, 0.6, (200, 200, 200), 1)
                cv2.imshow("Theft Detection - Behavioral AI", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            sv_dets = sv.Detections(
                xyxy=np.array(boxes_sv),
                confidence=np.array(confs_sv),
                class_id=np.array(cls_sv),
            )
            tracked = self.tracker.update_with_detections(sv_dets)

            all_persons = {}
            all_kps = {}
            used_canonicals = set()
            _embed_inputs  = []
            _color_hists   = {}
            _embeddings    = {}

            for i in range(len(tracked)):
                bbox = tracked.xyxy[i]
                tid = int(tracked.tracker_id[i])

                best_idx, best_sc = -1, -1.0
                bxc, byc = bbox_center(bbox)
                for idx, (rb, _) in persons_raw.items():
                    rx1, ry1, rx2, ry2 = rb
                    inter = max(0, min(rx2, bbox[2]) - max(rx1, bbox[0])) * max(0, min(ry2, bbox[3]) - max(ry1, bbox[1]))
                    union = (rx2 - rx1) * (ry2 - ry1) + (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) - inter + 1e-6
                    iou = inter / union
                    dist = np.hypot(bxc - bbox_center(rb)[0], byc - bbox_center(rb)[1]) / (bbox[2] - bbox[0] + 1e-6)
                    sc = iou - dist * 0.5
                    if sc > best_sc:
                        best_sc, best_idx = sc, idx

                kps_raw = persons_raw[best_idx][1] if best_sc > 0.1 else np.zeros((17, 3), dtype=np.float32)

                clean = np.copy(kps_raw)
                bx1, by1, bx2, by2 = bbox
                mx, my = (bx2 - bx1) * 0.15, (by2 - by1) * 0.15
                for k in range(17):
                    kx, ky, kc = clean[k]
                    if kc > 0.15 and (kx < bx1 - mx or kx > bx2 + mx or ky < by1 - my or ky > by2 + my):
                        clean[k][2] = 0.0

                _embed_inputs.append((tid, bbox))
                color_hist = self.appearance_extractor.color_hist(frame, bbox)
                _color_hists[tid] = color_hist
                embedding = _embeddings.get(tid)
                canonical_id, was_remapped, reid_rec = self.reid.resolve(
                    tid, embedding, _color_hists.get(tid), bbox, self.frame_idx, used_canonicals
                )

                if was_remapped:
                    self._transfer_state(tid, canonical_id)
                    if canonical_id in self.confirmed_thieves:
                        rec = self.confirmed_thieves[canonical_id]
                        self.reid.register(
                            canonical_id, embedding, color_hist, bbox, self.frame_idx,
                            confirmed=True, peak_risk=rec["peak_risk"], confirm_reasons=rec["reasons"]
                        )
                else:
                    is_c = canonical_id in self.confirmed_thieves
                    self.reid.register(
                        canonical_id, embedding, color_hist, bbox, self.frame_idx,
                        confirmed=is_c,
                        peak_risk=self.confirmed_thieves.get(canonical_id, {}).get("peak_risk", 0.0),
                        confirm_reasons=self.confirmed_thieves.get(canonical_id, {}).get("reasons", []),
                    )

                used_canonicals.add(canonical_id)
                all_persons[canonical_id] = bbox
                all_kps[canonical_id] = clean

            _embeddings = self.appearance_extractor.extract_batch(
                frame, _embed_inputs, self.frame_idx
            )

            confirmed_set = set(self.confirmed_thieves.keys())
            frame_risks = {}
            person_debug = {}

            self.object_tracker.update(self._last_obj_dets, all_kps, self.frame_idx)

            for pid, bbox in all_persons.items():
                kps = all_kps.get(pid, np.zeros((17, 3), dtype=np.float32))
                feats = self.extract_features(pid, bbox, kps, all_persons, confirmed_set, h, w)
                self.seq_buffer[pid].append(feats)

                rule_risk, reasons = self.compute_rule_based_risk(pid, feats)

                lstm_risk = 0.0
                if len(self.seq_buffer[pid]) == self.seq_length:
                    seq_t = torch.tensor(np.array(self.seq_buffer[pid], dtype=np.float32)).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        lstm_risk = self.classifier(seq_t).item()

                final_risk = 0.85 * rule_risk + 0.15 * lstm_risk

                obj_sig = self.object_tracker.get_signals(pid, kps, self.frame_idx)
                if obj_sig["hand_near_object"]:
                    bonus = 0.12 * obj_sig["object_grab_score"]
                    final_risk = min(final_risk + bonus, 1.0)
                    reasons.append(f"HandNearObject:{obj_sig['object_grab_score']:.2f}")
                if obj_sig["object_vanished"]:
                    final_risk = min(final_risk + 0.35, 1.0)
                    reasons.append("ObjectVanishedAfterTouch!")

                body_speed = feats[10]
                self.speed_window[pid].append(body_speed)
                avg_sp = float(np.mean(self.speed_window[pid]))
                sp_mult = 1.0 + min(avg_sp * 7.0, 0.45)
                if (feats[9] > 0.25 and feats[16] > 0.22) or (feats[6] > 0.25 and feats[7] > 0.18):
                    final_risk = min(final_risk * sp_mult, 1.0)
                    if sp_mult > 1.10 and final_risk > 0.45:
                        reasons.append(f"SpeedMult:x{sp_mult:.2f}")

                motion_evidence = (
                    (feats[16] > 0.42 and feats[9] > 0.22) or
                    (feats[16] > 0.38 and feats[8] > 0.22) or
                    (feats[16] > 0.38 and feats[7] > 0.22) or
                    (feats[8] > 0.26 and feats[16] > 0.22 and feats[9] > 0.18) or
                    (float(np.mean(self.grab_history[pid])) > 0.34
                     and float(np.max(self.wrist_speed_history[pid])) > 0.30
                     and float(np.mean(self.pull_history[pid])) > 0.18) or
                    (feats[6] > 0.22 and feats[7] > 0.22 and feats[3] > 0.22 and feats[9] > 0.25) or
                    (float(np.mean(self.lean_history[pid]))  > 0.20 and
                     float(np.mean(self.table_history[pid])) > 0.18 and
                     float(np.mean(self.grab_history[pid]))  > 0.22 and
                     len(self.grab_history[pid]) >= 4)
                )

                interaction_score = (
                    0.40 * feats[9] +
                    0.20 * feats[7] +
                    0.20 * feats[16] +
                    0.10 * feats[8] +
                    0.10 * feats[3]
                )

                frame_risks[pid] = (final_risk, reasons)
                person_debug[pid] = {
                    "bbox": bbox,
                    "feats": feats,
                    "motion_evidence": motion_evidence,
                    "interaction_score": interaction_score,
                    "suppressed": False,
                    "dominant_actor": True,
                }

            self.suppress_nearby_false_positives(frame_risks, person_debug, w)

            for pid, bbox in all_persons.items():
                final_risk, reasons = frame_risks.get(pid, (0.0, []))
                feats = person_debug[pid]["feats"]
                motion_evidence = person_debug[pid]["motion_evidence"]
                suppressed = person_debug[pid]["suppressed"]

                dominant_actor = person_debug[pid].get("dominant_actor", True)
                cooldown_active = self.near_confirm_cooldown.get(pid, 0) > 0

                self.confirm_buffer[pid].append(1 if (final_risk > 0.46 and motion_evidence and not suppressed and dominant_actor and not cooldown_active) else 0)
                self.cluster_dominance_buffer[pid].append(1 if dominant_actor and not suppressed else 0)

                recent_hits = int(sum(self.confirm_buffer[pid]))
                dominance_hits = int(sum(self.cluster_dominance_buffer[pid]))

                confirm_th = 0.54
                sticky_floor = 0.72
                temporal_confirm = (
                    recent_hits >= 4 and
                    len(self.confirm_buffer[pid]) >= 6 and
                    dominance_hits >= 4 and
                    motion_evidence and
                    not suppressed and
                    not cooldown_active
                )
                independent_motion = (
                    (feats[16] > 0.45 and feats[9] > 0.20) or
                    (feats[16] > 0.28 and feats[8] > 0.26 and feats[9] > 0.18) or
                    (feats[6] > 0.22 and feats[7] > 0.22 and feats[3] > 0.22 and feats[9] > 0.25) or
                    (feats[6] > 0.20 and feats[7] > 0.20 and feats[9] > 0.22)
                )

                presence_at_table = (
                    feats[7] > 0.15 or feats[11] > 0.25 or
                    (feats[6] > 0.20 and feats[7] > 0.15)
                )
                can_confirm = (
                    motion_evidence and
                    independent_motion and
                    not suppressed and
                    dominant_actor and
                    not cooldown_active and
                    final_risk >= confirm_th and
                    presence_at_table
                ) or (
                    temporal_confirm
                )

                if can_confirm and pid not in self.confirmed_thieves:
                    boosted_risk = max(final_risk, 0.68 if temporal_confirm else final_risk)
                    self.confirmed_thieves[pid] = {
                        "first_frame": self.frame_idx,
                        "reasons": list(reasons) + ([f"TemporalHits:{recent_hits}"] if temporal_confirm else []),
                        "peak_risk": boosted_risk,
                    }
                    self.reid.register(pid, None, None, bbox, self.frame_idx,
                                       confirmed=True, peak_risk=boosted_risk,
                                       confirm_reasons=list(reasons))
                    final_risk = boosted_risk

                if pid in self.confirmed_thieves and (suppressed or not dominant_actor or cooldown_active) and pid not in self.active_alarms:
                    first_frame = self.confirmed_thieves[pid].get("first_frame", self.frame_idx)
                    age = self.frame_idx - first_frame
                    if age <= 40:
                        del self.confirmed_thieves[pid]
                        final_risk = min(final_risk, 0.18)
                        reasons = list(reasons) + ["Revoked:ClusterFalsePositive"]

                if pid in self.confirmed_thieves:
                    final_risk = max(final_risk, sticky_floor)
                    ct = self.confirmed_thieves[pid]
                    if final_risk > ct["peak_risk"]:
                        ct["peak_risk"] = final_risk
                    reasons = ["[CONFIRMED THIEF]"] + ct["reasons"][:3]

                frame_risks[pid] = (final_risk, reasons)
                self.risk_scores[pid] = final_risk
                if final_risk > 0.55:
                    self.active_alarms[pid] = (60, reasons)
                self.prev_bbox[pid] = bbox

            frame_seg = self.seg_masks.get(self.frame_idx, {})
            pid_mask_map = {}
            for seg_pi, mask in frame_seg.items():
                if seg_pi in persons_raw:
                    seg_bbox = persons_raw[seg_pi][0]
                    best_pid, best_iou = None, 0.0
                    for pid2, pbbox in all_persons.items():
                        iou_val = bbox_iou(seg_bbox, pbbox)
                        if iou_val > best_iou:
                            best_iou, best_pid = iou_val, pid2
                    if best_pid is not None and best_iou > 0.35:
                        pid_mask_map[best_pid] = mask

            for pid, bbox in all_persons.items():
                kps = all_kps.get(pid, np.zeros((17, 3), dtype=np.float32))
                risk, reasons = frame_risks.get(pid, (0.0, []))
                seg_mask = pid_mask_map.get(pid)

                if pid in self.active_alarms and pid not in self.confirmed_thieves:
                    cnt, ar = self.active_alarms[pid]
                    if cnt > 0:
                        reasons = ar
                        risk = max(risk, 0.7)
                        self.active_alarms[pid] = (cnt - 1, ar)
                    else:
                        del self.active_alarms[pid]

                self.draw_person(frame, pid, bbox, kps, risk, reasons, seg_mask=seg_mask)

            self.draw_objects(frame, self._last_obj_dets)

            visible_ids = list(all_persons.keys())
            self.reid.end_frame(self.frame_idx, visible_ids)

            conf_vis = [p for p in all_persons if p in self.confirmed_thieves]
            high_risk = [p for p, (r, _) in frame_risks.items() if r > 0.65]

            if conf_vis or high_risk:
                ov = frame.copy()
                cv2.rectangle(ov, (0, 0), (w, 44), (0, 0, 160), -1)
                cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)
                ids = list(set(conf_vis + high_risk))
                label = "!!! CONFIRMED THIEF" if conf_vis else "!!! THEFT ALERT"
                cv2.putText(frame, f"{label} - IDs: {ids} !!!", (10, 30), self.font, 0.7, (0, 0, 255), 2)
                if self.confirmed_thieves:
                    cv2.putText(frame, f"Total confirmed: {len(self.confirmed_thieves)}", (w - 220, 30), self.font, 0.55, (0, 80, 255), 1)

            fps_txt = f"Frame:{self.frame_idx}  Persons:{len(all_persons)}  FPS:{self._fps_val:.1f}"
            cv2.putText(frame, fps_txt, (10, h - 10), self.font, 0.5, (180, 180, 180), 1)

            disp = frame
            if w > MAX_DISPLAY_W:
                disp = cv2.resize(frame, (MAX_DISPLAY_W, int(h * MAX_DISPLAY_W / w)))
            cv2.imshow("Theft Detection - Behavioral AI", disp)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            if self.frame_idx % 300 == 0:
                self.reid.cleanup(self.frame_idx)
                self.object_tracker.cleanup(self.frame_idx)
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Done.")

if __name__ == "__main__":
    VIDEO_PATH = r"E:\Computer science\python\Task\test_video.f137.mp4"
    TheftDetectionPipeline(VIDEO_PATH).run()