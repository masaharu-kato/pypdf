"""
Code related to text extraction.

Some parts are still in _page.py. In doubt, they will stay there.
"""

import copy
import math
from typing import Any, Callable, Optional, Union

from ..generic import DictionaryObject, TextStringObject, encode_pdfdocencoding

CUSTOM_RTL_MIN: int = -1
CUSTOM_RTL_MAX: int = -1
CUSTOM_RTL_SPECIAL_CHARS: list[int] = []
LAYOUT_NEW_BT_GROUP_SPACE_WIDTHS: int = 5


Mat = tuple[float, float, float, float, float, float]

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
    return (a*x + c*y + e, b*x + d*y + f)


class OrientationNotFoundError(Exception):
    pass


class CharMap:
    """
    (Added by Masaharu-Kato)
    Charactor-map data class for `extract_text` method in PageObject
    """
    def __init__(
        self,
        encoding: str | dict[int, str],
        map_dict: dict[str, str],
        font_res_name: str, # internal name, not the real font-name
        font_dict: dict | None, # The font-dictionary describes the font
    ):
        self.encoding = encoding
        self.map_dict = map_dict
        self.font_res_name = font_res_name
        self.font_dict = font_dict

    def __str__(self):
        return self.font_res_name
    
    def __repr__(self):
        return repr(self.font_dict)


class TextState:
    """
    (Added by Masaharu-Kato)
    Text state
    """
    def __init__(
        self,
        cm_matrix: Mat,
        tm_matrix: Mat,
        charmap: CharMap,
        font_size: float,
        char_scale: float,
        space_scale: float,
        _space_width: float,
        text_leading: float,
        text_offset: float,
        rtl_dir: bool, # right-to-left
    ):
        self.cm_matrix = cm_matrix
        self.tm_matrix = tm_matrix
        self.cmap = charmap
        self.font_size = font_size
        self.char_scale = char_scale
        self.space_scale = space_scale
        self._space_width = _space_width
        self.text_leading = text_leading
        self.text_offset = text_offset
        self.rtl_dir = rtl_dir  # right-to-left
    
    @property
    def space_width(self):
        return self._space_width / 1000.0

    def pos(self) -> tuple[float, float]:
        x, y = xy_mult((self.tm_matrix[4], self.tm_matrix[5]), self.cm_matrix)

        if self.text_offset:
            m = mult(self.tm_matrix, self.cm_matrix)
            k = math.sqrt(abs(m[0] * m[3]) + abs(m[1] * m[2]))
            x += self.text_offset * (self.font_size * k)

        return x, y
    
    
    def copy(self):
        return copy.copy(self)


def set_custom_rtl(
    _min: Union[str, int, None] = None,
    _max: Union[str, int, None] = None,
    specials: Union[str, list[int], None] = None,
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


def crlf_space_check(
    text: str,
    st: TextState,
    cmtm_prev: tuple[Mat, Mat],
    orientations: tuple[int, ...],
    output: str,
    processing_TJ_op: bool,
    visitor_text: Optional[Callable[[str, TextState], None]],
) -> tuple[str, str, Mat, Mat]:
    
    def push_text():
        nonlocal output, text
        output += text + "\n"
        if visitor_text is not None:
            visitor_text(text + "\n", st.copy())
        # if processing_TJ_op:
        #     st.text_offset += len(text + "\n")
        text = ""

    cm_prev = cmtm_prev[0]
    tm_prev = cmtm_prev[1]

    m_prev = mult(tm_prev, cm_prev)
    m = mult(st.tm_matrix, st.cm_matrix)
    orientation = orient(m)
    delta_x = m[4] - m_prev[4]
    delta_y = m[5] - m_prev[5]
    k = math.sqrt(abs(m[0] * m[3]) + abs(m[1] * m[2]))
    f = st.font_size * k
    cm_prev = m
    if orientation not in orientations:
        raise OrientationNotFoundError
    try:
        if orientation == 0:
            if delta_y < -0.8 * f:
                if (output + text)[-1] != "\n":
                    push_text()
            elif (
                abs(delta_y) < f * 0.3
                and abs(delta_x) > st.space_width * f * 15
                and (output + text)[-1] != " "
            ):
                text += " "
        elif orientation == 180:
            if delta_y > 0.8 * f:
                if (output + text)[-1] != "\n":
                    push_text()
            elif (
                abs(delta_y) < f * 0.3
                and abs(delta_x) > st.space_width * f * 15
                and (output + text)[-1] != " "
            ):
                text += " "
        elif orientation == 90:
            if delta_x > 0.8 * f:
                if (output + text)[-1] != "\n":
                    push_text()
            elif (
                abs(delta_x) < f * 0.3
                and abs(delta_y) > st.space_width * f * 15
                and (output + text)[-1] != " "
            ):
                text += " "
        elif orientation == 270:
            if delta_x < -0.8 * f:
                if (output + text)[-1] != "\n":
                    push_text()
            elif (
                abs(delta_x) < f * 0.3
                and abs(delta_y) > st.space_width * f * 15
                and (output + text)[-1] != " "
            ):
                text += " "
    except Exception:
        pass
    tm_prev = st.tm_matrix
    cm_prev = st.cm_matrix
    return text, output, cm_prev, tm_prev


def handle_tj(
    text: str,
    operands: list[Union[str, TextStringObject]],
    st: TextState,
    orientations: tuple[int, ...],
    output: str,
    processing_TJ_op: bool,
    visitor_text: Optional[Callable[[str, TextState], None]],
) -> str:
    
    def push_text():
        nonlocal output, text
        output += text
        if visitor_text is not None:
            visitor_text(text, st.copy())
        if processing_TJ_op:
            st.text_offset += len(text)
        text = ""

    m = mult(st.tm_matrix, st.cm_matrix)
    orientation = orient(m)
    if orientation in orientations and len(operands) > 0:
        if isinstance(operands[0], str):
            text += operands[0]
        else:
            t: str = ""
            tt: bytes = (
                encode_pdfdocencoding(operands[0])
                if isinstance(operands[0], str)
                else operands[0]
            )
            if isinstance(st.cmap.encoding, str):
                try:
                    t = tt.decode(st.cmap.encoding, "surrogatepass")  # apply str encoding
                except Exception:
                    # the data does not match the expectation,
                    # we use the alternative ;
                    # text extraction may not be good
                    t = tt.decode(
                        "utf-16-be" if st.cmap.encoding == "charmap" else "charmap",
                        "surrogatepass",
                    )  # apply str encoding
            else:  # apply dict encoding
                t = "".join(
                    [st.cmap.encoding[x] if x in st.cmap.encoding else bytes((x,)).decode() for x in tt]
                )
            # "\u0590 - \u08FF \uFB50 - \uFDFF"
            for x in [st.cmap.map_dict[x] if x in st.cmap.map_dict else x for x in t]:
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
                    text = x + text if st.rtl_dir else text + x
                elif (  # right-to-left characters set
                    0x0590 <= xx <= 0x08FF
                    or 0xFB1D <= xx <= 0xFDFF
                    or 0xFE70 <= xx <= 0xFEFF
                    or CUSTOM_RTL_MIN <= xx <= CUSTOM_RTL_MAX
                ):
                    if not st.rtl_dir:
                        st.rtl_dir = True
                        push_text()
                    text = x + text
                else:  # left-to-right
                    # print(">",xx,x,end="")
                    if st.rtl_dir:
                        st.rtl_dir = False
                        push_text()
                    text = text + x
                # fmt: on
    return text
