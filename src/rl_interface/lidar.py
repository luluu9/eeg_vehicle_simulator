
import numpy as np
import math

class SimulatedLidar:
    """
    Simulates Lidar sensors by calculating distance to track edges.
    Since WheelchairRacing environment might not have physical walls (only road tiles),
    we calculate intersection with the road boundaries defined by the track data.
    """
    def __init__(self, num_rays=7, fov=np.pi, max_range=150.0):
        self.num_rays = num_rays
        self.fov = fov
        self.max_range = max_range
        self.angles = np.linspace(-fov/2, fov/2, num_rays)
        
    def scan(self, env) -> np.ndarray:
        """
        Returns distances for each ray.
        Expects env to be a WheelchairRacing environment (or wrapper).
        """
        # Unwrap if needed to access internal attributes
        if hasattr(env, 'unwrapped'):
            env = env.unwrapped
            
        if not hasattr(env, 'car') or env.car is None:
            return np.full(self.num_rays, self.max_range)
            
        car_pos = env.car.hull.position # Box2D vector
        car_angle = env.car.hull.angle

        track = env.track
        if not track:
            return np.full(self.num_rays, self.max_range)
        
        closest_idx = self._find_closest_track_point(car_pos, track)
        
        # 4 Local segments: 2 back, 2 forward
        # Determine valid road polygons for these segments
        # Points (x,y) are center. Road width is 40/SCALE (approx 6.66 world units).
        TRACK_WIDTH = 40.0 / 6.0 
        
        # Check a reasonable window of segments to catch curves
        # +/- 6 segments covers enough distance for near-range lidar
        search_radius = 6 
        num_track = len(track)
        
        distances = []
        
        # Pre-calculate Ray directions
        rays = []
        # Car body is defined along Y-axis (Front is +Y).
        # Box2D 0-angle is +X.
        # So Car Forward is (car_angle + 90 deg).
        heading_offset = math.pi / 2
        
        for angle in self.angles:
            global_angle = car_angle + angle + heading_offset
            rays.append( (math.cos(global_angle), math.sin(global_angle)) )
        
        px, py = car_pos[0], car_pos[1]
        
        for rx, ry in rays:
            min_dist = self.max_range
            
            # Check segment walls
            for k in range(-search_radius, search_radius + 1):
                idx = (closest_idx + k) % num_track
                idx_prev = (idx - 1) % num_track
                
                # Geometry from environment
                # track[i] is current, track[i-1] is previous
                a1, b1, x1, y1 = track[idx]
                a2, b2, x2, y2 = track[idx_prev]
                
                # We need to construct the 2 wall segments for this track segment
                # Left Wall: (x1_l, y1_l) -> (x2_l, y2_l)
                # Right Wall: (x1_r, y1_r) -> (x2_r, y2_r)
                
                # Node 1
                cw1, sw1 = math.cos(b1), math.sin(b1)
                x1_l, y1_l = x1 - TRACK_WIDTH * cw1, y1 - TRACK_WIDTH * sw1
                x1_r, y1_r = x1 + TRACK_WIDTH * cw1, y1 + TRACK_WIDTH * sw1
                
                # Node 2
                cw2, sw2 = math.cos(b2), math.sin(b2)
                x2_l, y2_l = x2 - TRACK_WIDTH * cw2, y2 - TRACK_WIDTH * sw2
                x2_r, y2_r = x2 + TRACK_WIDTH * cw2, y2 + TRACK_WIDTH * sw2
                
                # Check intersection with Left Wall Segment
                d = self._ray_segment_intersect(px, py, rx, ry, x1_l, y1_l, x2_l, y2_l)
                if d is not None and d < min_dist:
                    min_dist = d
                    
                # Check intersection with Right Wall Segment
                d = self._ray_segment_intersect(px, py, rx, ry, x1_r, y1_r, x2_r, y2_r)
                if d is not None and d < min_dist:
                    min_dist = d
                    
            distances.append(min_dist)

        return np.array(distances)
        
    def _ray_segment_intersect(self, px, py, rx, ry, x1, y1, x2, y2):
        # Ray: P + t R
        # Segment: A + u (B - A)
        # P + t R = A + u S
        # t R - u S = A - P
        
        sx, sy = x2 - x1, y2 - y1
        
        # Cramers Rule or Cross Product 2D
        # | rx  -sx | | t | = | ax - px |
        # | ry  -sy | | u |   | ay - py |
        
        det = rx * (-sy) - ry * (-sx)
        if abs(det) < 1e-6:
            return None
            
        dx = x1 - px
        dy = y1 - py
        
        t = (dx * (-sy) - dy * (-sx)) / det
        u = (rx * dy - ry * dx) / det
        
        if t > 0 and 0.0 <= u <= 1.0:
            return t
        return None
        
    def _find_closest_track_point(self, pos, track):
        # Quick messy linear search
        # Optimization: Start search from expected index if we had state
        # But brute force 300 points is fast enough
        min_d2 = float('inf')
        best_idx = 0
        px, py = pos[0], pos[1]
        
        for i, (alpha, beta, tx, ty) in enumerate(track):
            d2 = (tx - px)**2 + (ty - py)**2
            if d2 < min_d2:
                min_d2 = d2
                best_idx = i
        return best_idx
