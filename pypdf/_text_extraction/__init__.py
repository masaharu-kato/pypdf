"""
Code related to text extraction.

Some parts are still in _page.py. In doubt, they will stay there.
"""

import math
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass

from ..generic import TextStringObject, encode_pdfdocencoding

W_CHAR_HAN = 0.67
W_CHAR_ZEN = W_CHAR_HAN * 2

CUSTOM_RTL_MIN: int = -1
CUSTOM_RTL_MAX: int = -1
CUSTOM_RTL_SPECIAL_CHARS: list[int] = []
LAYOUT_NEW_BT_GROUP_SPACE_WIDTHS: int = 5


Mat = tuple[float, float, float, float, float, float]  # Matrix [[a, b, 0], [c, d, 0], [e, f, 1]]

def mult(m: Mat, n: Mat) -> Mat:
    return (
        m[0] * n[0] + m[1] * n[2],
        m[0] * n[1] + m[1] * n[3],
        m[2] * n[0] + m[3] * n[2],
        m[2] * n[1] + m[3] * n[3],
        m[4] * n[0] + m[5] * n[2] + n[4],
        m[4] * n[1] + m[5] * n[3] + n[5],
    )

def xy_mult(xy: tuple[float, float], mat: Mat) -> tuple[float, float]:
    x, y = xy  # [[x, y]]
    a, b, c, d, e, f = mat # [[a, b, 0], [c, d, 0], [e, f, 1]]
    return (a*x + b*y + e, c*x + d*y + f)


class OrientationNotFoundError(Exception):
    pass


@dataclass
class CharMap:
    """
    (Added by Masaharu-Kato)
    Charactor-map data class for `extract_text` method in PageObject
    """
    encoding: str | dict[int, str]
    map_dict: dict[str, str]
    font_res_name: str # internal name, not the real font-name
    font_dict: dict | None # The font-dictionary describes the font

    def __str__(self):
        return self.font_res_name
    
    def __repr__(self):
        return repr(self.font_dict)


@dataclass
class TextState:
    """
    (Added by Masaharu-Kato)
    Text state
    """
    cm_matrix: Mat
    tm_matrix: Mat
    cmap: CharMap
    font_size: float
    char_scale: float  
    char_spacing: float
    space_scale: float  # 0.0 - 1.0
    _space_width: float
    text_leading: float
    box_left: float  # text-box left (x-offset)
    rtl_dir: bool # right-to-left
    

    @property
    def space_width(self):
        return self._space_width / 1000.0


class TextBoxData:
    def __init__(self, ts: TextState, text: str):
        
        # self._ts = ts
        self._text = text

        m = mult(ts.tm_matrix, ts.cm_matrix)

        tx, ty = 0.0, 0.0

        direction = -1.0 if ts.rtl_dir else 1.0
        tx += ts.box_left * direction

        # テキスト空間の座標 (tx, ty) を行列 m によってデバイス空間へ変換する
        self._x, self._y = xy_mult((tx, ty), m)
        
        # 行列の「スケール成分」を抽出してデバイス空間の w, h に変換
        # 行列 m から、X軸方向とY軸方向の純粋な拡大率（ベクトルの長さ）を計算します
        scale_x = math.sqrt(m[0] ** 2 + m[1] ** 2)
        scale_y = math.sqrt(m[2] ** 2 + m[3] ** 2)
        
        text_lines = self._text.split('\n')
        self._raw_w = _calc_box_width(ts, text_lines)
        self._raw_h = _calc_box_height(ts, len(text_lines))

        self._w = self._raw_w * scale_x
        self._h = self._raw_h * scale_y

        self._space_width = (W_CHAR_HAN * ts.font_size + 2 * ts.char_spacing + ts.space_scale) * ts.char_scale * scale_x
        self._space_height = (ts.font_size + 2 * ts.text_leading) * scale_y

    # @property
    # def text_state(self):
    #     return self._ts

    @property
    def text(self):
        return self._text

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def raw_w(self):
        return self._raw_w

    @property
    def raw_h(self):
        return self._raw_h

    @property
    def w(self):
        return self._w

    @property
    def h(self):
        return self._h
    
    @property
    def space_width(self):
        return self._space_width
    
    @property
    def space_height(self):
        return self._space_height
    
    def __repr__(self):
        return f'TextBoxData((x={self.x}, y={self.y}, w={self.w}, h={self.h}), "{self.text}")'



def _calc_box_width(ts: TextState, text_lines: list[str]):
    return max(_calc_line_text_size(ts, line) for line in text_lines)

def _calc_box_height(ts: TextState, n_lines: int):
    return (n_lines - 1) * abs(ts.text_leading) + ts.font_size

def _calc_line_text_size(ts: TextState, line_text: str):
    total_w = 0.0
    for i, ch in enumerate(line_text):

        char_w = W_CHAR_ZEN if unicodedata.east_asian_width(ch) in ('W', 'F', 'A') else W_CHAR_HAN
        total_w += char_w * ts.font_size
        
        # if i < len(line_text) - 1:
        total_w += ts.char_spacing
        # スペース文字（32）の後にのみ Tw を追加で適用
        if ch == ' ':
            total_w += ts.space_scale

    return total_w * ts.char_scale


def set_custom_rtl(
    _min: str | int | None = None,
    _max: str | int | None = None,
    specials: str | list[int] | None = None,
) -> tuple[int, int, list[int]]:
    """
    Change the Right-To-Left and special characters custom parameters.

    Args:
        _min: The new minimum value for the range of custom characters that
            will be written right to left.
            If set to ``None``, the value will not be changed.
            If set to an integer or string, it will be converted to its ASCII code.
            The default value is -1, which sets no additional range to be converted.
        _max: The new maximum value for the range of custom characters that will
            be written right to left.
            If set to ``None``, the value will not be changed.
            If set to an integer or string, it will be converted to its ASCII code.
            The default value is -1, which sets no additional range to be converted.
        specials: The new list of special characters to be inserted in the
            current insertion order.
            If set to ``None``, the current value will not be changed.
            If set to a string, it will be converted to a list of ASCII codes.
            The default value is an empty list.

    Returns:
        A tuple containing the new values for ``CUSTOM_RTL_MIN``,
        ``CUSTOM_RTL_MAX``, and ``CUSTOM_RTL_SPECIAL_CHARS``.
    """
    global CUSTOM_RTL_MIN, CUSTOM_RTL_MAX, CUSTOM_RTL_SPECIAL_CHARS
    if isinstance(_min, int):
        CUSTOM_RTL_MIN = _min
    elif isinstance(_min, str):
        CUSTOM_RTL_MIN = ord(_min)
    if isinstance(_max, int):
        CUSTOM_RTL_MAX = _max
    elif isinstance(_max, str):
        CUSTOM_RTL_MAX = ord(_max)
    if isinstance(specials, str):
        CUSTOM_RTL_SPECIAL_CHARS = [ord(x) for x in specials]
    elif isinstance(specials, list):
        CUSTOM_RTL_SPECIAL_CHARS = specials
    return CUSTOM_RTL_MIN, CUSTOM_RTL_MAX, CUSTOM_RTL_SPECIAL_CHARS


def orient(m: Mat) -> int:
    if m[3] > 1e-6:
        return 0
    elif m[3] < -1e-6:
        return 180
    elif m[1] > 0:
        return 90
    else:
        return 270


def handle_tj(
    # text: str,
    operands: list[str | bytes | TextStringObject],
    ts: TextState,
    orientations: tuple[int, ...],
    # output: str,
    # processing_TJ_op: bool,
    visitor_text: Callable[[TextBoxData], None] | None,
    verbose = False,
) -> str:
    
    def push_text(text: str):
        # nonlocal output, text
        if not text:
            return
        # output += text
        textbox = TextBoxData(ts, text)
        if verbose:
            print("    push_text (in handle_tj)", textbox)
        if visitor_text is not None:
            visitor_text(textbox)
        # if processing_TJ_op:
        #     st.box_left += textbox.raw_w
        # text = ""

    text = ""
    m = mult(ts.tm_matrix, ts.cm_matrix)
    orientation = orient(m)
    if orientation in orientations and len(operands) > 0:
        op0 = operands[0]
        if isinstance(op0, str):  # `TextStringObject` is instance of `str`
            text += op0  # op0: str | TextStringObject
        else:
            t: str = ""
            assert isinstance(op0, bytes)
            tt: bytes = op0
            if isinstance(ts.cmap.encoding, str):
                try:
                    t = tt.decode(ts.cmap.encoding, "surrogatepass")  # apply str encoding
                except Exception:
                    # the data does not match the expectation,
                    # we use the alternative ;
                    # text extraction may not be good
                    t = tt.decode(
                        "utf-16-be" if ts.cmap.encoding == "charmap" else "charmap",
                        "surrogatepass",
                    )  # apply str encoding
            else:  # apply dict encoding
                t = "".join(
                    [ts.cmap.encoding[x] if x in ts.cmap.encoding else bytes((x,)).decode() for x in tt]
                )
            # "\u0590 - \u08FF \uFB50 - \uFDFF"
            for x in [ts.cmap.map_dict[x] if x in ts.cmap.map_dict else x for x in t]:
                # x can be a sequence of bytes ; ex: habibi.pdf
                if len(x) == 1:
                    xx = ord(x)
                else:
                    xx = 1
                # fmt: off
                if (
                    # cases where the current inserting order is kept
                    (xx <= 0x2F)                        # punctuations but...
                    or 0x3A <= xx <= 0x40               # numbers (x30-39)
                    or 0x2000 <= xx <= 0x206F           # upper punctuations..
                    or 0x20A0 <= xx <= 0x21FF           # but (numbers) indices/exponents
                    or xx in CUSTOM_RTL_SPECIAL_CHARS   # customized....
                ):
                    text = x + text if ts.rtl_dir else text + x
                elif (  # right-to-left characters set
                    0x0590 <= xx <= 0x08FF
                    or 0xFB1D <= xx <= 0xFDFF
                    or 0xFE70 <= xx <= 0xFEFF
                    or CUSTOM_RTL_MIN <= xx <= CUSTOM_RTL_MAX
                ):
                    if not ts.rtl_dir:
                        ts.rtl_dir = True
                        push_text(text)
                        text = ""
                    text = x + text
                else:  # left-to-right
                    # print(">",xx,x,end="")
                    if ts.rtl_dir:
                        ts.rtl_dir = False
                        push_text(text)
                        text = ""
                    text = text + x
                # fmt: on
    return text
