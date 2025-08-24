import torch
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

class DetectorTracker:
    """
    Ultralytics YOLO (v8 API, can load yolov5s.pt too) + DeepSORT
    Detects COCO class 0 (person) and returns list of (track_id, (x,y,w,h)).
    """
    def __init__(self, conf_thresh=0.4, max_age=30, n_init=3, max_cosine_distance=0.3, weights="yolov5s.pt"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load YOLO model via ultralytics package. You can switch to 'yolov8n.pt' if preferred.
        self.model = YOLO(weights)  # downloads on first use
        self.conf_thresh = conf_thresh

        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_cosine_distance=max_cosine_distance
        )

    def detect_and_track(self, frame_bgr):
        """
        Args:
            frame_bgr: np.ndarray (H,W,3), BGR
        Returns:
            List[Tuple[int, (x,y,w,h)]]
        """
        # Run YOLO inference; classes=[0] filters to 'person'
        results = self.model.predict(
            source=frame_bgr,
            conf=self.conf_thresh,
            classes=[0],
            verbose=False
        )[0]

        dets = []
        if results.boxes is not None and len(results.boxes) > 0:
            xyxy = results.boxes.xyxy.cpu().numpy()
            conf = results.boxes.conf.cpu().numpy()
            cls  = results.boxes.cls.cpu().numpy()
            for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w = max(1, x2 - x1)
                h = max(1, y2 - y1)
                dets.append(([x1, y1, w, h], float(c), "person"))

        tracks = self.tracker.update_tracks(dets, frame=frame_bgr)
        out = []
        for t in tracks:
            if not t.is_confirmed():
                continue
            x, y, w, h = t.to_tlwh()
            out.append((t.track_id, (int(x), int(y), int(w), int(h))))
        return out
