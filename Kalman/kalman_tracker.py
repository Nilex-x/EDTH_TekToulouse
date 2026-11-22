"""
Tracker multi-objets avec filtre de Kalman (inspiré de SORT / ByteTrack).

Principe, image par image :
    1. PRÉDIRE  : chaque piste devine où son objet sera dans l'image actuelle.
    2. ASSOCIER : on relie les détections du modèle aux positions PRÉDITES
                  (IoU + algorithme hongrois, puis distance entre centres
                  pour les petits objets qui ne se recouvrent plus).
    3. CORRIGER : les pistes associées sont corrigées avec leur détection.
    4. GÉRER LA VIE DES PISTES :
         - une piste non vue est gardée `max_age` images (patience) ;
         - une nouvelle piste doit être vue `min_hits` fois avant d'être
           "confirmée" (prudence contre les fausses détections).

Format des boîtes partout : [x, y, w, h] en pixels (coin haut-gauche + taille),
comme dans les fichiers MOT et dans predict_onnx_baseline.py.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


class _SimpleKalmanFilter:
    """Drop-in pure-NumPy replacement for ``filterpy.kalman.KalmanFilter``."""

    def __init__(self, dim_x: int, dim_z: int) -> None:
        self.x = np.zeros((dim_x, 1), dtype=np.float64)
        self.P = np.eye(dim_x, dtype=np.float64)
        self.F = np.eye(dim_x, dtype=np.float64)
        self.H = np.zeros((dim_z, dim_x), dtype=np.float64)
        self.Q = np.eye(dim_x, dtype=np.float64)
        self.R = np.eye(dim_z, dtype=np.float64)
        self._I = np.eye(dim_x, dtype=np.float64)

    def predict(self) -> None:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z: np.ndarray) -> None:
        z_col = np.asarray(z, dtype=np.float64).reshape(-1, 1)
        y = z_col - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = np.linalg.solve(S, self.H @ self.P).T
        self.x = self.x + K @ y
        self.P = (self._I - K @ self.H) @ self.P

# Une détection = (class_id 0-based, score, [x, y, w, h])
# C'est exactement le format produit par _nms_dets() dans predict_onnx_baseline.py
Detection = tuple[int, float, list[float]]


# =============================================================================
# 1. Petits outils sur les boîtes
# =============================================================================


def xywh_to_cxcywh(box) -> np.ndarray:
    """[x, y, w, h] (coin haut-gauche) -> [cx, cy, w, h] (centre)."""
    x, y, w, h = box
    return np.array([x + w / 2.0, y + h / 2.0, w, h], dtype=np.float64)


def cxcywh_to_xywh(state) -> np.ndarray:
    """[cx, cy, w, h] (centre) -> [x, y, w, h] (coin haut-gauche)."""
    cx, cy, w, h = state[:4]
    return np.array([cx - w / 2.0, cy - h / 2.0, w, h], dtype=np.float64)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU entre toutes les boîtes de `a` (N,4) et de `b` (M,4), format xywh."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax1, ay1 = a[:, 0:1], a[:, 1:2]
    ax2, ay2 = ax1 + a[:, 2:3], ay1 + a[:, 3:4]
    bx1, by1 = b[:, 0], b[:, 1]
    bx2, by2 = bx1 + b[:, 2], by1 + b[:, 3]
    inter_w = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    inter_h = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    inter = inter_w * inter_h
    union = a[:, 2:3] * a[:, 3:4] + b[:, 2] * b[:, 3] - inter
    return inter / np.maximum(union, 1e-9)


def center_distance_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Distance entre les centres, divisée par la taille moyenne des deux boîtes.
    Une valeur de 1.0 = "les centres sont éloignés d'environ une taille de boîte".
    Utile pour les tout petits objets : leur IoU tombe vite à 0 dès qu'ils bougent.
    """
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ca = a[:, :2] + a[:, 2:] / 2.0
    cb = b[:, :2] + b[:, 2:] / 2.0
    dist = np.linalg.norm(ca[:, None, :] - cb[None, :, :], axis=2)
    size_a = np.sqrt(a[:, 2] * a[:, 3])[:, None]
    size_b = np.sqrt(b[:, 2] * b[:, 3])[None, :]
    return dist / np.maximum((size_a + size_b) / 2.0, 1e-9)


# =============================================================================
# 2. Réglages du tracker (les "boutons" à tester)
# =============================================================================


@dataclass
class TrackerConfig:
    # --- Gestion de la vie des pistes ---
    max_age: int = 15  # patience : nb d'images qu'on garde une piste non vue
    min_hits: int = 3  # prudence : nb de détections avant de confirmer une piste
    max_age_tentative: int = 2  # une piste pas encore confirmée meurt plus vite

    # --- Association ---
    iou_threshold: float = 0.2  # IoU minimum pour relier une détection à une piste
    center_threshold: float = 1.5  # 2e passe : distance de centres max (en "tailles de boîte")

    # --- Bruits du filtre de Kalman (proportionnels à la taille de la boîte) ---
    # Q : à quel point l'objet peut changer de trajectoire
    std_weight_position: float = 1.0 / 20.0
    std_weight_velocity: float = 1.0 / 160.0
    # R : à quel point le détecteur est imprécis
    std_weight_measurement: float = 1.0 / 20.0
    min_std_px: float = 1.0  # plancher en pixels (sinon bruit ~0 pour les boîtes minuscules)

    # --- Sortie ---
    interpolate_gaps: bool = True  # boucher les trous d'une piste par interpolation linéaire


# =============================================================================
# 3. Une piste = un objet suivi, avec son propre filtre de Kalman
# =============================================================================


class KalmanBoxTracker:
    """
    Suit UN objet. État du filtre (8 valeurs) :
        x = [cx, cy, w, h, vx, vy, vw, vh]
    Le détecteur ne mesure que [cx, cy, w, h] ; les vitesses sont déduites
    automatiquement par le filtre au fil des images.
    """

    def __init__(self, det: Detection, track_id: int, frame: int, cfg: TrackerConfig):
        class_id, score, box = det
        self.cfg = cfg
        self.id = track_id

        # --- 3a. Création du filtre : 8 valeurs d'état, 4 valeurs mesurées ---
        self.kf = _SimpleKalmanFilter(dim_x=8, dim_z=4)

        # F (transition) : modèle "vitesse constante"
        #   nouvelle_position = position + vitesse   (dt = 1 image)
        self.kf.F = np.eye(8)
        for i in range(4):
            self.kf.F[i, i + 4] = 1.0

        # H (observation) : le détecteur voit seulement [cx, cy, w, h]
        self.kf.H = np.eye(4, 8)

        # État initial : position = la détection, vitesse = 0
        self.kf.x[:4, 0] = xywh_to_cxcywh(box)

        # P (incertitude initiale) : on connaît la position, PAS la vitesse
        w, h = box[2], box[3]
        pos_std = self._std(self.cfg.std_weight_position * 2, w, h)
        vel_std = self._std(self.cfg.std_weight_velocity * 10, w, h)
        self.kf.P = np.diag(np.concatenate([pos_std, vel_std]) ** 2)

        # --- 3b. Informations de suivi ---
        self.hits = 1  # nb de fois où l'objet a été détecté
        self.time_since_update = 0  # nb d'images depuis la dernière détection
        self.confirmed = cfg.min_hits <= 1
        self.class_votes: Counter[int] = Counter({class_id: 1})
        self.last_score = score
        self.last_box = np.asarray(box, dtype=np.float64)  # dernière boîte VUE
        self.last_frame = frame

    def _std(self, weight: float, w: float, h: float) -> np.ndarray:
        """Écart-type proportionnel à la taille : [w, h, w, h] * poids, avec un plancher."""
        std = weight * np.array([w, h, w, h], dtype=np.float64)
        return np.maximum(std, self.cfg.min_std_px)

    # ---- PRÉDIRE ----
    def predict(self) -> np.ndarray:
        """Avance l'état d'une image et renvoie la boîte prédite [x, y, w, h]."""
        # Empêche la largeur / hauteur de devenir négatives
        for i in (2, 3):
            if self.kf.x[i, 0] + self.kf.x[i + 4, 0] <= 1.0:
                self.kf.x[i + 4, 0] = 0.0

        # Q recalculé à chaque image selon la taille actuelle de la boîte
        w, h = self.kf.x[2, 0], self.kf.x[3, 0]
        q_pos = self._std(self.cfg.std_weight_position, w, h)
        q_vel = self._std(self.cfg.std_weight_velocity, w, h)
        self.kf.Q = np.diag(np.concatenate([q_pos, q_vel]) ** 2)

        self.kf.predict()  # x = F·x  et  P = F·P·Fᵀ + Q
        self.kf.x[2, 0] = max(self.kf.x[2, 0], 1.0)
        self.kf.x[3, 0] = max(self.kf.x[3, 0], 1.0)
        self.time_since_update += 1
        return self.box

    # ---- CORRIGER ----
    def update(self, det: Detection, frame: int) -> None:
        """Corrige l'état avec une détection associée."""
        class_id, score, box = det
        w, h = box[2], box[3]
        # R recalculé selon la taille de la détection
        self.kf.R = np.diag(self._std(self.cfg.std_weight_measurement, w, h) ** 2)
        self.kf.update(xywh_to_cxcywh(box))  # gain K, puis correction de x et P

        self.hits += 1
        self.time_since_update = 0
        if self.hits >= self.cfg.min_hits:
            self.confirmed = True
        self.class_votes[class_id] += 1
        self.last_score = score
        self.last_box = np.asarray(box, dtype=np.float64)
        self.last_frame = frame

    @property
    def box(self) -> np.ndarray:
        """Boîte actuelle estimée par le filtre, format [x, y, w, h]."""
        return cxcywh_to_xywh(self.kf.x[:4, 0])

    @property
    def class_id(self) -> int:
        """Classe la plus souvent détectée pour cette piste (vote)."""
        return self.class_votes.most_common(1)[0][0]


# =============================================================================
# 4. Le tracker : gère toutes les pistes d'une vidéo
# =============================================================================


class KalmanTracker:
    """
    Utilisation :
        tracker = KalmanTracker()
        for frame, dets in enumerate(detections_par_image, start=1):
            tracker.step(frame, dets)
        rows = tracker.finish()          # toutes les lignes MOT
        write_mot_file("seq_0001.txt", rows)
    """

    def __init__(self, cfg: TrackerConfig | None = None):
        self.cfg = cfg or TrackerConfig()
        self.tracks: list[KalmanBoxTracker] = []
        self.next_id = 1
        self.results: list[tuple] = []  # lignes des pistes confirmées
        self.pending: dict[int, list[tuple]] = {}  # lignes des pistes pas encore confirmées

    # ---- Une image ----
    def step(self, frame: int, detections: list[Detection]) -> None:
        # --- Étape 1 : PRÉDIRE la position de toutes les pistes ---
        predicted = np.array([t.predict() for t in self.tracks]).reshape(-1, 4)
        det_boxes = np.array([d[2] for d in detections], dtype=np.float64).reshape(-1, 4)

        # --- Étape 2 : ASSOCIER détections <-> positions prédites ---
        matches, unmatched_dets = self._associate(predicted, det_boxes)

        # --- Étape 3 : CORRIGER les pistes associées ---
        for t_idx, d_idx in matches:
            track = self.tracks[t_idx]
            gap_start = track.last_frame
            gap_box = track.last_box
            track.update(detections[d_idx], frame)
            self._record_gap(track, gap_start, gap_box, frame)
            self._record(track, frame, track.last_box, track.last_score)

        # --- Étape 4a : créer une piste pour chaque détection non associée ---
        for d_idx in unmatched_dets:
            track = KalmanBoxTracker(detections[d_idx], self.next_id, frame, self.cfg)
            self.next_id += 1
            self.tracks.append(track)
            self._record(track, frame, track.last_box, track.last_score)

        # --- Étape 4b : supprimer les pistes perdues depuis trop longtemps ---
        alive = []
        for track in self.tracks:
            limit = self.cfg.max_age if track.confirmed else self.cfg.max_age_tentative
            if track.time_since_update > limit:
                self.pending.pop(track.id, None)  # jamais confirmée -> on oublie ses lignes
            else:
                alive.append(track)
        self.tracks = alive

    def finish(self) -> list[tuple]:
        """Fin de la vidéo : renvoie toutes les lignes, triées par image puis par id."""
        return sorted(self.results, key=lambda r: (r[0], r[1]))

    # ---- Association en 2 passes ----
    def _associate(self, trk_boxes: np.ndarray, det_boxes: np.ndarray):
        n_trk, n_det = len(trk_boxes), len(det_boxes)
        if n_trk == 0 or n_det == 0:
            return [], list(range(n_det))

        matches: list[tuple[int, int]] = []
        BIG = 1e6

        # Passe 1 : IoU entre boîtes PRÉDITES et détections (algorithme hongrois)
        ious = iou_matrix(trk_boxes, det_boxes)
        cost = np.where(ious >= self.cfg.iou_threshold, 1.0 - ious, BIG)
        for r, c in zip(*linear_sum_assignment(cost)):
            if cost[r, c] < BIG:
                matches.append((r, c))

        # Passe 2 : pour les restes, distance entre centres (petits objets rapides)
        um_trk = [i for i in range(n_trk) if i not in {m[0] for m in matches}]
        um_det = [j for j in range(n_det) if j not in {m[1] for m in matches}]
        if um_trk and um_det:
            dist = center_distance_matrix(trk_boxes[um_trk], det_boxes[um_det])
            cost2 = np.where(dist <= self.cfg.center_threshold, dist, BIG)
            for r, c in zip(*linear_sum_assignment(cost2)):
                if cost2[r, c] < BIG:
                    matches.append((um_trk[r], um_det[c]))

        matched_dets = {m[1] for m in matches}
        unmatched_dets = [j for j in range(n_det) if j not in matched_dets]
        return matches, unmatched_dets

    # ---- Enregistrement des lignes de sortie ----
    def _record(self, track: KalmanBoxTracker, frame: int, box, score: float) -> None:
        """Ajoute une ligne. Une piste non confirmée met ses lignes en attente."""
        row = (frame, track.id, *[float(v) for v in box], float(score))
        if track.confirmed:
            # Si la piste vient d'être confirmée, on publie aussi ses lignes en attente
            for old in self.pending.pop(track.id, []):
                self.results.append(old + (track.class_id,))
            self.results.append(row + (track.class_id,))
        else:
            self.pending.setdefault(track.id, []).append(row)

    def _record_gap(self, track, last_frame: int, last_box, frame: int) -> None:
        """Bouche un trou (objet raté quelques images) par interpolation linéaire."""
        if not self.cfg.interpolate_gaps or frame - last_frame <= 1:
            return
        new_box = track.last_box
        for f in range(last_frame + 1, frame):
            a = (f - last_frame) / (frame - last_frame)
            box = (1 - a) * last_box + a * new_box
            self._record(track, f, box, track.last_score * 0.5)


# =============================================================================
# 5. Écriture au format MOTChallenge attendu par evaluate_submissions.py
# =============================================================================


def write_mot_file(path: Path | str, rows: list[tuple]) -> None:
    """
    Une ligne par boîte :
        frame (1-based), track_id, x, y, w, h, score, class_id (1-based), 1.0
    """
    lines = []
    for frame, tid, x, y, w, h, score, cls in rows:
        if w <= 0 or h <= 0:
            continue  # le correcteur refuse les boîtes invalides
        lines.append(
            f"{frame},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{score:.4f},{cls + 1},1.0"
        )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
