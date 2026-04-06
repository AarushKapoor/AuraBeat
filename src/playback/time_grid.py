# ============================================================
#  TIME GRID (time → pixels)
# ============================================================

class TimeGrid:
    """
    Converts time (seconds) into vertical pixel positions.
    Controls:
        - pixels_per_second (scroll speed)
        - zooming
        - total song length → canvas height
    """

    def __init__(self, pixels_per_second=120.0):
        # 120 px/sec = 2 seconds per 240px, feels good for Synthesia-like flow
        self.px_per_sec = float(pixels_per_second)

    # -----------------------------
    # Time → Y pixel
    # -----------------------------
    def time_to_y(self, t_seconds: float) -> float:
        return t_seconds * self.px_per_sec

    # -----------------------------
    # Y pixel → Time
    # -----------------------------
    def y_to_time(self, y_pixels: float) -> float:
        return y_pixels / self.px_per_sec

    # -----------------------------
    # Duration → Height
    # -----------------------------
    def duration_to_height(self, duration: float) -> float:
        return duration * self.px_per_sec

    # -----------------------------
    # Compute canvas height
    # -----------------------------
    def compute_canvas_height(self, total_time: float) -> float:
        return max(2000, total_time * self.px_per_sec)
