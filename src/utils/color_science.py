"""
Color Science Utilities

Implements OkLCH color space conversions and accessibility helpers.
Reference: https://bottosson.github.io/posts/oklab/
"""

import math
from typing import Tuple

def srgb_transfer_function(a: float) -> float:
    """Convert sRGB component to linear light."""
    return ((a + 0.055) / 1.055) ** 2.4 if a >= 0.04045 else a / 12.92

def srgb_transfer_function_inv(a: float) -> float:
    """Convert linear light component to sRGB."""
    return 1.055 * (a ** (1 / 2.4)) - 0.055 if a >= 0.0031308 else 12.92 * a

def linear_srgb_to_oklab(r: float, g: float, b: float) -> Tuple[float, float, float]:
    """Convert linear sRGB to Oklab."""
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b

    l_ = l ** (1/3)
    m_ = m ** (1/3)
    s_ = s ** (1/3)

    L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    b_ = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_

    return L, a, b_

def oklab_to_linear_srgb(L: float, a: float, b: float) -> Tuple[float, float, float]:
    """Convert Oklab to linear sRGB."""
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b

    l = l_ ** 3
    m = m_ ** 3
    s = s_ ** 3

    r = +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    b_out = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s

    return r, g, b_out

def oklab_to_oklch(L: float, a: float, b: float) -> Tuple[float, float, float]:
    """Convert Oklab to OkLCH."""
    C = math.sqrt(a * a + b * b)
    h = math.atan2(b, a) * (180 / math.pi)
    if h < 0:
        h += 360
    return L, C, h

def oklch_to_oklab(L: float, C: float, h: float) -> Tuple[float, float, float]:
    """Convert OkLCH to Oklab."""
    h_rad = h * (math.pi / 180)
    a = C * math.cos(h_rad)
    b = C * math.sin(h_rad)
    return L, a, b

def srgb_to_oklch(r: float, g: float, b: float) -> Tuple[float, float, float]:
    """Convert sRGB (0-1) to OkLCH."""
    r_lin = srgb_transfer_function(r)
    g_lin = srgb_transfer_function(g)
    b_lin = srgb_transfer_function(b)

    L, a, b_ = linear_srgb_to_oklab(r_lin, g_lin, b_lin)
    return oklab_to_oklch(L, a, b_)

def hex_to_oklch(hex_color: str) -> Tuple[float, float, float]:
    """Convert Hex to OkLCH."""
    hex_color = hex_color.lstrip('#')
    r_int, g_int, b_int = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    return srgb_to_oklch(r_int / 255.0, g_int / 255.0, b_int / 255.0)

def oklch_to_hex(L: float, C: float, h: float) -> str:
    """Convert OkLCH to Hex."""
    L_lab, a_lab, b_lab = oklch_to_oklab(L, C, h)
    r_lin, g_lin, b_lin = oklab_to_linear_srgb(L_lab, a_lab, b_lab)

    r = max(0, min(1, srgb_transfer_function_inv(r_lin)))
    g = max(0, min(1, srgb_transfer_function_inv(g_lin)))
    b = max(0, min(1, srgb_transfer_function_inv(b_lin)))

    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

def calculate_luminance(r: float, g: float, b: float) -> float:
    """Calculate relative luminance for WCAG contrast."""
    # Input is 0-1 sRGB
    R = srgb_transfer_function(r)
    G = srgb_transfer_function(g)
    B = srgb_transfer_function(b)
    return 0.2126 * R + 0.7152 * G + 0.0722 * B

def calculate_contrast_ratio(hex1: str, hex2: str) -> float:
    """Calculate WCAG 2.1 contrast ratio between two hex colors."""
    hex1 = hex1.lstrip('#')
    r1, g1, b1 = tuple(int(hex1[i:i+2], 16) / 255.0 for i in (0, 2, 4))

    hex2 = hex2.lstrip('#')
    r2, g2, b2 = tuple(int(hex2[i:i+2], 16) / 255.0 for i in (0, 2, 4))

    l1 = calculate_luminance(r1, g1, b1)
    l2 = calculate_luminance(r2, g2, b2)

    lighter = max(l1, l2)
    darker = min(l1, l2)

    return (lighter + 0.05) / (darker + 0.05)
