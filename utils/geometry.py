"""Pure 2D/3D geometry helpers — no library dependencies."""

import math


def midpoint(p1, p2):
    """Return midpoint of two (x, y) or (x, y, z) tuples."""
    if len(p1) == 3:
        return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2, (p1[2] + p2[2]) / 2)
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)


def euclidean_distance(p1, p2):
    """Euclidean distance between two points (2D or 3D)."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


def normalize_vector(v):
    """Return unit vector of v."""
    mag = math.sqrt(sum(c ** 2 for c in v))
    if mag < 1e-9:
        return tuple(0.0 for _ in v)
    return tuple(c / mag for c in v)


def dot_product(v1, v2):
    return sum(a * b for a, b in zip(v1, v2))


def angle_between(v1, v2):
    """Angle in degrees between two vectors (2D or 3D)."""
    n1 = normalize_vector(v1)
    n2 = normalize_vector(v2)
    cos_val = dot_product(n1, n2)
    # Clamp to avoid floating-point domain errors in acos
    cos_val = max(-1.0, min(1.0, cos_val))
    return math.degrees(math.acos(cos_val))


def angle_at_vertex(a, b, c):
    """Angle at point b formed by the vectors b->a and b->c (degrees)."""
    v1 = tuple(a[i] - b[i] for i in range(len(a)))
    v2 = tuple(c[i] - b[i] for i in range(len(c)))
    return angle_between(v1, v2)


def vector_angle_from_horizontal(p1, p2):
    """Angle in degrees of the vector p1->p2 measured from positive x-axis."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.degrees(math.atan2(dy, dx))


def vector_angle_from_vertical(p1, p2):
    """Angle in degrees of the vector p1->p2 measured from positive y-axis (downward)."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    # Angle from vertical (0,1)
    return angle_between((dx, dy), (0, 1))
