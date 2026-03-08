import logging
#logging.basicConfig(filename='debug.log', encoding='utf-8',level=logging.DEBUG)
import flet as ft
from typing import List, Dict, Tuple
import sys
import os
import numpy as np
from PIL import Image,ImageGrab,ImageDraw
from pathlib import Path
sys.path.append(os.path.join(".","src"))
import ocr
from tools.ndlkoten2tei import convert_tei
import xml.etree.ElementTree as ET
import time
from concurrent.futures import ThreadPoolExecutor
import time
import json
import shutil
import argparse
import yaml
import io
import glob
import pypdfium2
import base64
import ctypes
import re
import unicodedata
from io import BytesIO
from uicomponent.localelabel import TRANSLATIONS


from reading_order.xy_cut.eval import eval_xml
from ndl_parser import convert_to_xml_string3
from ndl_parser import categories_org_name_index



name = "NDLOCR-Lite-GUI"

PDFTMPPATH="4ab7ecc3-53fb-b3e7-64e8-a809b5a483d2"

def get_windows_scale_factor():
    try:
        ctypes.windll.user32.SetProcessDPIAware()
        hdc = ctypes.windll.user32.GetDC(0)
        dpi = ctypes.windll.gdi32.GetDeviceCaps(hdc, 88)
        ctypes.windll.user32.ReleaseDC(0, hdc)
        return dpi / 96.0  # 讓呎ｺ悶′96dpi縺ｪ縺ｮ縺ｧ縲√◎繧後ｒ蜑ｲ縺｣縺ｦ蛟咲紫繧貞・縺・
    except:
        return 1.0


class RecogLine:
    def __init__(self,npimg:np.ndarray,idx:float,pred_char_cnt:int,pred_str:str=""):
        self.npimg = npimg
        self.idx   = idx
        self.pred_char_cnt = pred_char_cnt
        self.pred_str = pred_str
    def __lt__(self, other):  
        return self.idx < other.idx


def _normalize_receipt_line(line: str) -> str:
    return unicodedata.normalize("NFKC", line).strip()


def _extract_datetime_from_lines(lines: List[str]) -> str:
    datetime_patterns = [
        r"(\d{4}[\/\.\-年]\d{1,2}[\/\.\-月]\d{1,2}(?:日)?\s*\d{1,2}:\d{2}(?::\d{2})?)",
        r"(\d{4}[\/\.\-年]\d{1,2}[\/\.\-月]\d{1,2}(?:日)?)",
        r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\s*\d{1,2}:\d{2})",
        r"(\d{1,2}:\d{2}(?::\d{2})?)",
    ]
    for line in lines:
        for pattern in datetime_patterns:
            m = re.search(pattern, line)
            if m:
                return m.group(1)
    return ""


def _extract_amount_candidates(line: str) -> List[str]:
    normalized = _normalize_receipt_line(line)
    money_pattern = r"(?:[¥￥$]\s*)?-?\d[\d,]*(?:\.\d{1,2})?"
    return re.findall(money_pattern, normalized)


def _normalize_amount(amount_text: str) -> str:
    cleaned = _normalize_receipt_line(amount_text).replace("¥", "").replace("￥", "").replace("$", "").replace(",", "").strip()
    return cleaned


def _bbox_bounds(item: Dict) -> Tuple[float, float, float, float]:
    bbox = item.get("boundingBox", [])
    if not bbox or len(bbox) < 4:
        return (0.0, 0.0, 0.0, 0.0)
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    return (float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys)))


def _select_total_with_coordinates(line_items: List[Dict]) -> Tuple[str, List[int], List[int]]:
    if not line_items:
        return "", [], []

    def is_amount_like_text(text: str) -> bool:
        t = _normalize_receipt_line(text)
        return bool(re.search(r"\d", t)) and bool(re.search(r"[¥￥$\d]", t))

    def union_bbox(items: List[Dict]) -> List[int]:
        if not items:
            return []
        xs1, ys1, xs2, ys2 = [], [], [], []
        for item in items:
            l, t, r, b = _bbox_bounds(item)
            xs1.append(l); ys1.append(t); xs2.append(r); ys2.append(b)
        return [int(min(xs1)), int(min(ys1)), int(max(xs2)), int(max(ys2))]

    prepared = []
    for item in line_items:
        text = _normalize_receipt_line(str(item.get("text", "")))
        if not text:
            continue
        l, t, r, b = _bbox_bounds(item)
        prepared.append({"item": item, "text": text, "l": l, "t": t, "r": r, "b": b, "cy": (t + b) / 2.0, "h": max(1.0, b - t)})
    prepared.sort(key=lambda x: x["cy"])

    groups: List[List[Dict]] = []
    for p in prepared:
        if not groups:
            groups.append([p])
            continue
        g = groups[-1]
        g_cy = sum(x["cy"] for x in g) / len(g)
        g_h = max(x["h"] for x in g)
        if abs(p["cy"] - g_cy) <= max(12.0, max(g_h, p["h"]) * 0.8):
            g.append(p)
        else:
            groups.append([p])
    for g in groups:
        g.sort(key=lambda x: x["l"])

    def line_has_total_label(group: List[Dict]) -> bool:
        joined = _normalize_receipt_line("".join(x["text"] for x in group).replace(" ", "")).lower()
        # Exclude discount-related totals (e.g. 値引合計 / 割引合計).
        exclude_keywords = [
            "\u5024\u5f15",       # 値引
            "\u5024\u5f15\u304d", # 値引き
            "\u5272\u5f15",       # 割引
            "discount",
            "coupon",
        ]
        if any(k in joined for k in exclude_keywords):
            return False
        keywords = [
            "\u5408\u8a08",       # 合計
            "\u7a0e\u5408\u8a08", # 税合計
            "\u7dcf\u5408\u8a08", # 総合計
            "total", "grandtotal", "amountdue",
        ]
        if any(k in joined for k in keywords):
            return True
        chars = {x["text"] for x in group}
        return ("\u5408" in chars and "\u8a08" in chars)

    def extract_amount_from_group(group: List[Dict], label_right: float) -> Tuple[str, List[int]]:
        candidates = [x for x in group if x["l"] >= label_right - 4 and is_amount_like_text(x["text"])]
        if not candidates:
            return "", []
        candidates.sort(key=lambda x: x["l"])
        clusters: List[List[Dict]] = []
        for x in candidates:
            if not clusters:
                clusters.append([x])
                continue
            prev = clusters[-1][-1]
            gap = x["l"] - prev["r"]
            if gap <= max(24.0, max(prev["h"], x["h"]) * 0.9):
                clusters[-1].append(x)
            else:
                clusters.append([x])

        best_amount = ""
        best_bbox = []
        best_right = -1.0
        for cl in clusters:
            merged = _normalize_receipt_line("".join(x["text"] for x in cl))
            amts = _extract_amount_candidates(merged)
            if not amts:
                continue
            amt = _normalize_amount(amts[-1])
            bbox = union_bbox([x["item"] for x in cl])
            right = max(x["r"] for x in cl)
            if right > best_right:
                best_right = right
                best_amount = amt
                best_bbox = bbox
        return best_amount, best_bbox

    best_amount = ""
    best_label_bbox: List[int] = []
    best_amount_bbox: List[int] = []
    best_score = None

    for g in groups:
        if not line_has_total_label(g):
            continue
        first_amount_idx = None
        for i, x in enumerate(g):
            if is_amount_like_text(x["text"]):
                first_amount_idx = i
                break
        label_items = [x for i, x in enumerate(g) if first_amount_idx is None or i < first_amount_idx]
        if not label_items:
            label_items = [x for x in g if not is_amount_like_text(x["text"])]
        label_right = max((x["r"] for x in label_items), default=min(x["l"] for x in g))

        amount, amount_bbox = extract_amount_from_group(g, label_right)
        if not amount:
            continue
        label_bbox = union_bbox([x["item"] for x in label_items]) if label_items else []

        score = (amount_bbox[1] if amount_bbox else 0) * 1000 + (amount_bbox[2] if amount_bbox else 0)
        if best_score is None or score > best_score:
            best_score = score
            best_amount = amount
            best_label_bbox = label_bbox
            best_amount_bbox = amount_bbox

    return best_amount, best_label_bbox, best_amount_bbox


def _find_line_item_bbox(line_items: List[Dict], target_text: str) -> List[int]:
    if not target_text:
        return []
    target = _normalize_receipt_line(target_text)
    for item in line_items:
        text = _normalize_receipt_line(str(item.get("text", "")))
        if text == target:
            l, t, r, b = _bbox_bounds(item)
            return [int(l), int(t), int(r), int(b)]
    for item in line_items:
        text = _normalize_receipt_line(str(item.get("text", "")))
        if target in text or text in target:
            l, t, r, b = _bbox_bounds(item)
            return [int(l), int(t), int(r), int(b)]
    return []


def _extract_datetime_with_bbox(lines: List[str], line_items: List[Dict]) -> Tuple[str, List[int]]:
    datetime_patterns = [
        r"(\d{4}[\/\.\-年]\d{1,2}[\/\.\-月]\d{1,2}(?:日)?\s*\d{1,2}:\d{2}(?::\d{2})?)",
        r"(\d{4}[\/\.\-年]\d{1,2}[\/\.\-月]\d{1,2}(?:日)?)",
        r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\s*\d{1,2}:\d{2})",
        r"(\d{1,2}:\d{2}(?::\d{2})?)",
    ]
    for line in lines:
        for pattern in datetime_patterns:
            m = re.search(pattern, line)
            if not m:
                continue
            dt_text = m.group(1)
            for item in line_items:
                text = _normalize_receipt_line(str(item.get("text", "")))
                if dt_text in text:
                    l, t, r, b = _bbox_bounds(item)
                    return dt_text, [int(l), int(t), int(r), int(b)]
            return dt_text, []
    return "", []


def extract_receipt_fields(text_lines: List[str], line_items: List[Dict] = None) -> Dict[str, str]:
    try:
        if line_items is None:
            line_items = []
        lines = [_normalize_receipt_line(line) for line in text_lines if _normalize_receipt_line(line)]
        if not lines:
            return {"store_name": "", "datetime": "", "total": "", "store_name_bbox": [], "datetime_bbox": [], "total_bbox": [], "total_label_bbox": []}

        store_name = ""
        for line in lines[:8]:
            if re.search(r"(tel|電話|領収|レシート|receipt|担当|no\.?|#)", line, flags=re.IGNORECASE):
                continue
            if len(re.sub(r"[\d\W_]+", "", line)) < 2:
                continue
            store_name = line
            break

        dt_value, dt_bbox = _extract_datetime_with_bbox(lines, line_items)
        if not dt_value:
            dt_value = _extract_datetime_from_lines(lines)
        total, total_label_bbox, total_bbox = _select_total_with_coordinates(line_items)
        store_bbox = _find_line_item_bbox(line_items, store_name)

        return {
            "store_name": store_name,
            "datetime": dt_value,
            "total": total,
            "store_name_bbox": store_bbox,
            "datetime_bbox": dt_bbox,
            "total_bbox": total_bbox,
            "total_label_bbox": total_label_bbox,
        }
    except Exception:
        # 繝ｬ繧ｷ繝ｼ繝域歓蜃ｺ螟ｱ謨玲凾繧０CR譛ｬ菴薙・邯咏ｶ壹＆縺帙ｋ
        return {"store_name": "", "datetime": "", "total": "", "store_name_bbox": [], "datetime_bbox": [], "total_bbox": [], "total_label_bbox": []}


def format_receipt_fields(fields: Dict[str, str], langcode: str = "ja") -> str:
    if langcode == "ja":
        title = "[レシート抽出]"
        labels = [("店舗名", "store_name"), ("日時", "datetime"), ("合計", "total")]
        na = "未検出"
    else:
        title = "[Receipt Extraction]"
        labels = [("Store", "store_name"), ("Datetime", "datetime"), ("Total", "total")]
        na = "N/A"
    body = [f"{label}: {fields.get(key) or na}" for label, key in labels]
    return "\n".join([title] + body)


def _draw_field_highlight(input_image_path: str, out_image_path: str, bboxes: List[List[int]]) -> bool:
    try:
        if not os.path.exists(input_image_path):
            return False
        valid_boxes = [b for b in bboxes if isinstance(b, list) and len(b) == 4]
        if not valid_boxes:
            return False
        im = Image.open(input_image_path).convert("RGB")
        draw = ImageDraw.Draw(im)
        for i, box in enumerate(valid_boxes):
            x1, y1, x2, y2 = box
            color = (255, 0, 0) if i == 0 else (0, 170, 255)
            width = 4 if i == 0 else 3
            draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        os.makedirs(os.path.dirname(out_image_path), exist_ok=True)
        im.save(out_image_path)
        return True
    except Exception:
        return False


def _draw_field_focus_view(input_image_path: str, out_image_path: str, target_bbox: List[int], aux_bboxes: List[List[int]] = None) -> bool:
    try:
        if not os.path.exists(input_image_path):
            return False
        if not target_bbox or len(target_bbox) != 4:
            return False
        im = Image.open(input_image_path).convert("RGB")
        x1, y1, x2, y2 = target_bbox
        focus_boxes = [target_bbox]
        if aux_bboxes:
            for box in aux_bboxes:
                if isinstance(box, list) and len(box) == 4:
                    focus_boxes.append(box)
        fx1 = min(b[0] for b in focus_boxes)
        fy1 = min(b[1] for b in focus_boxes)
        fx2 = max(b[2] for b in focus_boxes)
        fy2 = max(b[3] for b in focus_boxes)
        w, h = im.size
        cx = max(0, min(w, (fx1 + fx2) // 2))
        cy = max(0, min(h, (fy1 + fy2) // 2))

        # Zoomed focus view around clicked field.
        box_w = max(20, fx2 - fx1)
        box_h = max(20, fy2 - fy1)
        view_w = max(260, int(max(w * 0.50, box_w * 2.8)))
        view_h = max(260, int(max(h * 0.50, box_h * 4.0)))
        left = max(0, cx - view_w // 2)
        top = max(0, cy - view_h // 2)
        right = min(w, left + view_w)
        bottom = min(h, top + view_h)
        left = max(0, right - view_w)
        top = max(0, bottom - view_h)

        crop = im.crop((left, top, right, bottom))
        draw = ImageDraw.Draw(crop)
        draw.rectangle([x1 - left, y1 - top, x2 - left, y2 - top], outline=(255, 0, 0), width=5)

        if aux_bboxes:
            for box in aux_bboxes:
                if not box or len(box) != 4:
                    continue
                ax1, ay1, ax2, ay2 = box
                draw.rectangle([ax1 - left, ay1 - top, ax2 - left, ay2 - top], outline=(0, 170, 255), width=3)

        os.makedirs(os.path.dirname(out_image_path), exist_ok=True)
        crop.save(out_image_path)
        return True
    except Exception:
        return False
    
def process_cascade(alllineobj:RecogLine,recognizer30,recognizer50,recognizer100,is_cascade=True):
    targetdflist30,targetdflist50,targetdflist100,targetdflist200=[],[],[],[]
    for lineobj in alllineobj:
        if lineobj.pred_char_cnt==3 and is_cascade:
            targetdflist30.append(lineobj)
        elif lineobj.pred_char_cnt==2 and is_cascade:
            targetdflist50.append(lineobj)
        else:
            targetdflist100.append(lineobj)
    targetdflistall=[]
    with ThreadPoolExecutor(thread_name_prefix="thread") as executor:
        resultlines30,resultlines50,resultlines100,resultlines200=[],[],[],[]
        if len(targetdflist30)>0:
            resultlines30 = executor.map(recognizer30.read, [t.npimg for t in targetdflist30])
            resultlines30 = list(resultlines30)
        for i in range(len(targetdflist30)):
            pred_str=resultlines30[i]
            lineobj=targetdflist30[i]
            if len(pred_str)>=25:
                targetdflist50.append(lineobj)
            else:
                lineobj.pred_str=pred_str
                targetdflistall.append(lineobj)
        if len(targetdflist50)>0:
            resultlines50 = executor.map(recognizer50.read, [t.npimg for t in targetdflist50])
            resultlines50 = list(resultlines50)
        for i in range(len(targetdflist50)):
            pred_str=resultlines50[i]
            lineobj=targetdflist50[i]
            if len(pred_str)>=45:
                targetdflist100.append(lineobj)
            else:
                lineobj.pred_str=pred_str
                targetdflistall.append(lineobj)
        if len(targetdflist100)>0:
            resultlines100 = executor.map(recognizer100.read, [t.npimg for t in targetdflist100])
            resultlines100 = list(resultlines100)
        for i in range(len(targetdflist100)):
            pred_str=resultlines100[i]
            lineobj=targetdflist100[i]
            lineobj.pred_str=pred_str
            if len(pred_str)>=98 and lineobj.npimg.shape[0]<lineobj.npimg.shape[1]:
                baseimg=lineobj.npimg
                tmplineobj_1=RecogLine(npimg=baseimg[:,:baseimg.shape[1]//2,:],idx=lineobj.idx,pred_char_cnt=100)
                tmplineobj_2=RecogLine(npimg=baseimg[:,baseimg.shape[1]//2:,:],idx=lineobj.idx,pred_char_cnt=100)
                targetdflist200.append(tmplineobj_1)
                targetdflist200.append(tmplineobj_2)
            else:
                targetdflistall.append(lineobj)
        if len(targetdflist200)>0:
            resultlines200 = executor.map(recognizer100.read, [t.npimg for t in targetdflist200])
            resultlines200 = list(resultlines200)
            for i in range(0,len(targetdflist200)-1,2):
                ia=targetdflist200[i]
                lineobj=RecogLine(npimg=None,idx=ia.idx,pred_char_cnt=100,pred_str=resultlines200[i]+resultlines200[i+1])
                targetdflistall.append(lineobj)
        targetdflistall=sorted(targetdflistall)
        resultlinesall=[t.pred_str for t in targetdflistall]
    return resultlinesall



class ImageSelector:
    def __init__(self, page: ft.Page,config_obj:Dict,detector=None, recognizer30=None,recognizer50=None,recognizer100=None,outputdirpath=None,width: int =600, height: int = 600):
        self.cnt=0#繧ｯ繝ｭ繝・・譎ゅ・菫晏ｭ倡判蜒上・騾壹＠逡ｪ蜿ｷ逕ｨ
        self.page = page
        self.config_obj=config_obj
        self.langcode=config_obj["langcode"]
        self.inputpathlist=[]
        self.outputdirpath=outputdirpath
        
        self.image_src = "dummy.dat"
        self.dialog_width = width
        self.dialog_height = height
        self.page_index=0
        self.detector=detector
        self.recognizer30=recognizer30
        self.recognizer50=recognizer50
        self.recognizer100=recognizer100

        # 繝峨Λ繝・げ髢句ｧ倶ｽ咲ｽｮ繧剃ｿ晄戟縺吶ｋ螟画焚
        self.start_x = 0
        self.start_y = 0

        # 驕ｸ謚樒洸蠖｢逕ｨ縺ｮ Container・亥・譛溽憾諷九・蟷・・鬮倥＆0・・
        self.selection_box = ft.Container(
            left=0,
            top=0,
            width=0,
            height=0,
            border=ft.border.all(2, ft.Colors.BLUE),
            bgcolor=ft.Colors.TRANSPARENT,
        )

        # 逕ｻ蜒上・荳翫↓驟咲ｽｮ縺吶ｋ騾乗・縺ｪ繝ｬ繧､繝､繝ｼ・医ラ繝ｩ繝・げ謫堺ｽ懊・讀懃衍逕ｨ・・
        self.overlay = ft.GestureDetector(
            content=ft.Container(
                width=self.dialog_width,
                height=self.dialog_height,
                bgcolor=ft.Colors.TRANSPARENT,
            ),
            on_pan_start=self.pan_start,
            on_pan_update=self.pan_update,
            on_pan_end=self.pan_end,
        )
        self.img=ft.Image(src=self.image_src, width=self.dialog_width, height=self.dialog_height,fit=ft.ImageFit.CONTAIN)
        self.imgzm=ft.Image(src=self.image_src, width=self.dialog_width, height=self.dialog_height,fit=ft.ImageFit.CONTAIN)
        # Stack 繧ｦ繧｣繧ｸ繧ｧ繝・ヨ縺ｧ逕ｻ蜒上・∈謚樒洸蠖｢縲√が繝ｼ繝舌・繝ｬ繧､繧帝㍾縺ｭ繧・
        self.image_stack = ft.Stack(
            width=self.dialog_width,
            height=self.dialog_height,
            controls=[
                self.img,
                self.selection_box,
                self.overlay,
            ]
        )
        self.cropocr_btn=ft.ElevatedButton(TRANSLATIONS["imageselector_cropocr_btn"][self.langcode], on_click=self.crop_region)
        self.dialog = ft.AlertDialog(
            modal=True,
            content=self.image_stack,
            actions=[
                ft.ElevatedButton(TRANSLATIONS["imageselector_zoom_btn"][self.langcode],icon=ft.Icons.ZOOM_IN, on_click=self.open_zoom_page),
                ft.ElevatedButton(TRANSLATIONS["imageselector_prev_btn"][self.langcode], on_click=self.prev_page),
                ft.ElevatedButton(TRANSLATIONS["imageselector_next_btn"][self.langcode], on_click=self.next_page),
                self.cropocr_btn,
                ft.ElevatedButton(TRANSLATIONS["common_cancel"][self.langcode], on_click=self.close_dialog)
            ],
        )
        zoom_img=ft.InteractiveViewer(
            min_scale=1,
            max_scale=10,
            boundary_margin=ft.margin.all(20),
            content=self.imgzm)
        
        self.zoom_dialog=ft.AlertDialog(
            modal=True,
            content=zoom_img,
            actions=[
                ft.ElevatedButton(TRANSLATIONS["common_cancel"][self.langcode], on_click=self.close_zoom_page)
            ]
        )
        self.resulttext=ft.Text(value="",selectable=True,color=ft.Colors.BLACK)
        
        self.crop_image=ft.Image(src=self.image_src, width=300, height=300,fit=ft.ImageFit.CONTAIN)
        crop_image_col = ft.Column(
            controls=[self.crop_image],
            width=300,
            height=300,
            expand=False
        )
        self.crop_image_int=ft.InteractiveViewer(
            min_scale=1,
            max_scale=5,
            boundary_margin=ft.margin.all(20),
            content=crop_image_col)
        self.result_text_col = ft.Column(
            controls=[self.resulttext],
            scroll=ft.ScrollMode.ALWAYS,
            width=800,
            height=300,
            expand=False
        )
        
        self.result_dialog= ft.AlertDialog(
            title=ft.Text(TRANSLATIONS["imageselector_result_title"][self.langcode]),
            modal=True,
            content=ft.Row([self.crop_image_int,self.result_text_col]),
            actions=[
                ft.ElevatedButton("OK", on_click=self.close_result_page)
            ]
        )
    def open_result_page(self):
        self.dialog.open = False
        self.result_dialog.open = True
        self.page.overlay.append(self.result_dialog)
        self.page.update()

    def close_result_page(self,e):
        self.result_dialog.open = False
        self.dialog.open = True
        self.page.update()

    def set_image(self, inputpathlist):
        """逕ｻ蜒上た繝ｼ繧ｹ繧定ｨｭ螳壹☆繧九Γ繧ｽ繝・ラ"""
        self.cnt=0
        self.inputpathlist=inputpathlist
        self.image_src=inputpathlist[self.page_index]
        self.img.src = inputpathlist[self.page_index]
        self.imgzm.src = inputpathlist[self.page_index]
        self.page.update()

    def set_outputdir(self,outputdirpath):
        self.outputdirpath=outputdirpath


    def open_zoom_page(self,e):
        self.dialog.open = False
        if not self.zoom_dialog in self.page.overlay:
            self.page.overlay.append(self.zoom_dialog)
        self.zoom_dialog.open = True
        self.page.update()

    def close_zoom_page(self, e):
        self.zoom_dialog.open = False
        self.dialog.open = True
        self.page.update()

    # 繝峨Λ繝・げ髢句ｧ区凾・夐幕蟋句ｺｧ讓吶ｒ險倬鹸縺励・∈謚樒洸蠖｢繧貞・譛溷喧
    def pan_start(self, e: ft.DragStartEvent):
        self.start_x = e.local_x
        self.start_y = e.local_y
        self.selection_box.left = self.start_x
        self.selection_box.top = self.start_y
        self.selection_box.width = 0
        self.selection_box.height = 0
        self.page.update()

    # 繝峨Λ繝・げ荳ｭ・夐幕蟋倶ｽ咲ｽｮ縺ｨ迴ｾ蝨ｨ菴咲ｽｮ縺九ｉ遏ｩ蠖｢縺ｮ菴咲ｽｮ縺ｨ繧ｵ繧､繧ｺ繧定ｨ育ｮ励＠縺ｦ譖ｴ譁ｰ
    def pan_update(self, e: ft.DragUpdateEvent):
        cur_x, cur_y = e.local_x, e.local_y
        left = min(self.start_x, cur_x)
        top = min(self.start_y, cur_y)
        width = abs(cur_x - self.start_x)
        height = abs(cur_y - self.start_y)
        self.selection_box.left = left
        self.selection_box.top = top
        self.selection_box.width = width
        self.selection_box.height = height
        self.page.update()

    # 繝峨Λ繝・げ邨ゆｺ・凾・壽怙邨ら噪縺ｪ驕ｸ謚樣伜沺縺檎｢ｺ螳・
    def pan_end(self, e: ft.DragEndEvent):
        self.page.update()

    # 繝繧､繧｢繝ｭ繧ｰ繧偵・繝ｼ繧ｸ荳翫↓陦ｨ遉ｺ縺吶ｋ繝｡繧ｽ繝・ラ
    def open_dialog(self, e):
        self.page.overlay.append(self.dialog)
        self.dialog.open = True
        self.page.update()
    
    def prev_page(self, e):
        if self.page_index > 0:
            self.page_index -= 1
        else:
            self.page_index = len(self.inputpathlist) - 1
        self.img.src = self.inputpathlist[self.page_index]
        self.imgzm.src=self.inputpathlist[self.page_index]
        self.page.update()

    def next_page(self, e):
        if self.page_index < len(self.inputpathlist) - 1:
            self.page_index += 1
        else:
            self.page_index = 0
        self.img.src = self.inputpathlist[self.page_index]
        self.imgzm.src=self.inputpathlist[self.page_index]
        self.page.update()

    def crop_region(self, e):
        #print(self.image_src)
        pilimg=Image.open(self.img.src)
        pilimg=pilimg.convert('RGB')
        rwidth,rheight=pilimg.size
        if rheight<rwidth:
            window_h=self.dialog_height*rheight/rwidth
            window_w=self.dialog_width
            offset_h=(window_w-window_h)/2
            offset_w=0
        else:
            window_h=self.dialog_height
            window_w=self.dialog_width*rwidth/rheight
            offset_w=(window_h-window_w)/2
            offset_h=0
        hratio=rheight/window_h
        wratio=rwidth/window_w
        cropx=int((self.selection_box.left-offset_w)*wratio)
        cropy=int((self.selection_box.top-offset_h)*hratio)
        cropw=int(self.selection_box.width*wratio)
        croph=int(self.selection_box.height*hratio)
        if cropx>0 and cropy>0 and cropw>10 and croph>0:
            im_crop = pilimg.crop((cropx, cropy, cropx+cropw, cropy+croph))
        else:
            #im_crop = pilimg
            return
        buff = BytesIO()
        im_crop.save(buff, "png")
        self.crop_image.src_base64=base64.b64encode(buff.getvalue()).decode("utf-8")
        self.outputcroppedpath=os.path.join(os.getcwd(),PDFTMPPATH,os.path.basename(self.image_src).split(".")[0]+"_cropped_{}.jpg".format(self.cnt))
        #im_crop.save(self.outputcroppedpath)
        self.mini_ocr(im_crop)
        self.cnt+=1
        self.page.update()
    
    def mini_ocr(self,im_crop):
        self.cropocr_btn.disabled=True
        self.page.update()
        inputname=os.path.basename(self.outputcroppedpath)
        #print(inputname)
        
        tatelinecnt=0
        alllinecnt=0
        self.crop_image.src=im_crop
        npimg = np.array(im_crop)
        img_h,img_w=npimg.shape[:2]
        detections,classeslist=ocr.process_detector(detector=self.detector,inputname=inputname,npimage=npimg,outputpath=self.outputdirpath,issaveimg=False)
        #print(detections)
        resultobj=[dict(),dict()]
        resultobj[0][0]=list()
        for i in range(17):
            resultobj[1][i]=[]
        for det in detections:
            xmin,ymin,xmax,ymax=det["box"]
            conf=det["confidence"]
            if det["class_index"]==0:
                resultobj[0][0].append([xmin,ymin,xmax,ymax])
            resultobj[1][det["class_index"]].append([xmin,ymin,xmax,ymax,conf])
        xmlstr=convert_to_xml_string3(img_w, img_h, inputname, classeslist, resultobj)
        xmlstr="<OCRDATASET>"+xmlstr+"</OCRDATASET>"

        root = ET.fromstring(xmlstr)
        eval_xml(root, logger=None)
        alllineobj=[]
        alltextlist=[]
        for idx,lineobj in enumerate(root.findall(".//LINE")):
            xmin=int(lineobj.get("X"))
            ymin=int(lineobj.get("Y"))
            line_w=int(lineobj.get("WIDTH"))
            line_h=int(lineobj.get("HEIGHT"))
            try:
                pred_char_cnt=float(lineobj.get("PRED_CHAR_CNT"))
            except:
                pred_char_cnt=100.0
            if line_h>line_w:
                tatelinecnt+=1
            alllinecnt+=1
            lineimg=npimg[ymin:ymin+line_h,xmin:xmin+line_w,:]
            linerecogobj = RecogLine(lineimg,idx,pred_char_cnt)
            alllineobj.append(linerecogobj)

        resultlines=process_cascade(alllineobj,self.recognizer30,self.recognizer50,self.recognizer100,is_cascade=True)
        resultlines=list(resultlines)
        alltextlist.append("\n".join(resultlines))
        resjsonarray=[]
        for idx,lineobj in enumerate(root.findall(".//LINE")):
            lineobj.set("STRING",resultlines[idx])
            xmin=int(lineobj.get("X"))
            ymin=int(lineobj.get("Y"))
            line_w=int(lineobj.get("WIDTH"))
            line_h=int(lineobj.get("HEIGHT"))
            try:
                conf=float(lineobj.get("CONF"))
            except:
                conf=0
            jsonobj={"boundingBox": [[xmin,ymin],[xmin,ymin+line_h],[xmin+line_w,ymin],[xmin+line_w,ymin+line_h]],
                "id": idx,"isVertical": "true","text": resultlines[idx],"isTextline": "true","confidence": conf}
            resjsonarray.append(jsonobj)
        receipt_fields = extract_receipt_fields(resultlines, line_items=resjsonarray)
        receipt_summary = format_receipt_fields(receipt_fields, self.langcode)
        
        if alllinecnt==0 or tatelinecnt/alllinecnt>0.5:
            alltextlist=alltextlist[::-1]
        with open(os.path.join(self.outputdirpath,os.path.basename(inputname).split(".")[0]+".txt"),"w",encoding="utf-8") as wtf:
            wtf.write("\n".join(alltextlist))
        self.resulttext.value=receipt_summary + "\n\n" + "\n".join(alltextlist)
        self.cropocr_btn.disabled=False
        self.open_result_page()
        self.page.update()

    def close_dialog(self, e):
        self.dialog.open=False
        self.page.update()


class CaptureTool:
    def __init__(self, page: ft.Page,config_obj:Dict, detector=None, recognizer30=None, recognizer50=None, recognizer100=None, width: int = 400, height: int = 400):
        self.page = page
        self.config_obj=config_obj
        self.langcode=config_obj["langcode"]
        self.detector = detector
        self.recognizer30 = recognizer30
        self.recognizer50 = recognizer50
        self.recognizer100 = recognizer100
        self.dialog_width = width
        self.dialog_height = height
        self.im_crop=None
        self.img_str=""
        self.result_jsonstr=""
        self.outputdirpath = os.getcwd()
        """
        :param page: Flet縺ｮ繝壹・繧ｸ繧ｪ繝悶ず繧ｧ繧ｯ繝・
        """
        self.scale_factor = get_windows_scale_factor() # 霑ｽ蜉: 蛻晄悄蛹匁凾縺ｫ蛟咲紫繧貞叙蠕励＠縺ｦ縺翫￥
        # 驕ｸ謚樒ｯ・峇縺ｮ蠎ｧ讓・
        self.start_x = 0
        self.start_y = 0
        self.current_x = 0
        self.current_y = 0

        # 蜈・・繧ｦ繧｣繝ｳ繝峨え迥ｶ諷九ｒ菫晏ｭ倥☆繧句､画焚
        self.original_width = 0
        self.original_height = 0
        self.original_left = 0
        self.original_top = 0
        self.original_bgcolor = None

        # 驕ｸ謚樒ｯ・峇繧定｡ｨ遉ｺ縺吶ｋ繧ｳ繝ｳ繝・リ・域怙蛻昴・髱櫁｡ｨ遉ｺ・・
        self.selection_box = ft.Container(
            border=ft.border.all(2, ft.Colors.RED),
            bgcolor=ft.Colors.with_opacity(0.2, ft.Colors.RED),
            visible=False,
        )
        # 譏蜒剰｡ｨ遉ｺ逕ｨImage繧ｳ繝ｳ繝医Ο繝ｼ繝ｫ
        self.img_control = ft.Image(
            src_base64=None,
            src=None,
            width=self.dialog_width,
            height=self.dialog_height,
            fit=ft.ImageFit.CONTAIN,
            gapless_playback=True # 繧ｹ繝医Μ繝ｼ繝溘Φ繧ｰ譎ゅ・縺｡繧峨▽縺埼亟豁｢
        )
        self.retry_btn=ft.ElevatedButton(TRANSLATIONS["capturetool_retry_btn"][self.langcode], on_click=self.start_capture)
        self.cboc_fixed = ft.Checkbox(label=TRANSLATIONS["capturetool_fixregion"][self.langcode], value=False,disabled=True)
        self.ocr_btn = ft.ElevatedButton(TRANSLATIONS["capturetool_ocr_button"][self.langcode], on_click=self.mini_ocr)
        
        self.errorlog=ft.Text("")
        # 繝｡繧､繝ｳ繝繧､繧｢繝ｭ繧ｰ縺ｮ讒区・
        self.dialog_content = ft.Column(
            controls=[
                self.errorlog,
                ft.Container(
                    content=self.img_control,
                    border=ft.border.all(1, ft.Colors.GREY),
                    alignment=ft.alignment.center
                )
            ],
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            tight=True
        )

        self.dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text(TRANSLATIONS["capturetool_result_title"][self.langcode]),
            content=self.dialog_content,
            actions=[
                ft.Row([self.retry_btn,
                self.cboc_fixed,
                self.ocr_btn,
                ft.ElevatedButton(TRANSLATIONS["common_close"][self.langcode], on_click=self.close_dialog)])
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )
        self.resulttext = ft.Text(value="", selectable=True, color=ft.Colors.BLACK)
        self.resultsmessage=ft.Text(value="", selectable=True, color=ft.Colors.BLACK)
        self.llmstatus_text = ft.Text(value="", selectable=True, color=ft.Colors.BLACK)
        self.result_crop_image = ft.Image(src="", width=300, height=300, fit=ft.ImageFit.CONTAIN)
        
        self.crop_image_int = ft.InteractiveViewer(
            min_scale=1,
            max_scale=5,
            boundary_margin=ft.margin.all(20),
            content=ft.Column([self.result_crop_image], width=300, height=300)
        )
        
        self.result_text_col = ft.Column(
            controls=[self.resulttext],
            scroll=ft.ScrollMode.ALWAYS,
            width=600,
            height=300,
        )
        self.result_dialog = ft.AlertDialog(
            title=ft.Text(TRANSLATIONS["capturetool_resultocr_title"][self.langcode]),
            modal=True,
            content=ft.Row(
                controls=[self.crop_image_int, self.result_text_col],
                alignment=ft.MainAxisAlignment.START,
                vertical_alignment=ft.CrossAxisAlignment.START
            ),
            actions=[
                self.resultsmessage,
                self.llmstatus_text,
                #self.promptbtn,
                #self.llmbtn,
                ft.ElevatedButton(TRANSLATIONS["common_close"][self.langcode], on_click=self.close_result_page)
            ]
        )
        self.bibinfo_dialog = ft.AlertDialog(
            title=ft.Text("書誌情報"),
            content="",
            actions=[
                ft.TextButton("閉じる", on_click=self.close_bibinfo_page)
            ],
        )
        # 逕ｻ髱｢蜈ｨ菴薙ｒ隕・≧縺溘ａ縺ｮStack
        # GestureDetector縺ｧ繝峨Λ繝・げ謫堺ｽ懊ｒ讀懃衍縺吶ｋ
        self.overlay_stack = ft.Stack(
            controls=[
                ft.GestureDetector(
                    on_pan_start=self._on_pan_start,
                    on_pan_update=self._on_pan_update,
                    on_pan_end=self._on_pan_end,
                    drag_interval=10,
                ),
                self.selection_box,
            ],
            expand=True,
            visible=False, # 譛蛻昴・髱櫁｡ｨ遉ｺ
        )

        # 繝壹・繧ｸ譛蜑埼擇縺ｮ繧ｪ繝ｼ繝舌・繝ｬ繧､縺ｫ霑ｽ蜉縺励※縺翫￥
        self.page.overlay.append(self.overlay_stack)


    def start_capture(self, e=None):
        """Start capture mode."""
        if self.dialog.open:
            self.close_dialog(e)
        if self.cboc_fixed.value:
            self._capture_and_restore(self.x1_phys, self.y1_phys, self.x2_phys, self.y2_phys)
            return
        self.scale_factor = get_windows_scale_factor()
        self.original_width = self.page.window.width
        self.original_height = self.page.window.height
        self.original_left = self.page.window.left
        self.original_top = self.page.window.top
        self.original_bgcolor = self.page.bgcolor

        # 繧ｦ繧｣繝ｳ繝峨え繧貞・逕ｻ髱｢繝ｻ騾乗・繝ｻ譛蜑埼擇縺ｫ險ｭ螳・
        self.page.window.maximized = True
        #self.page.window.frameless = True
        self.page.window.title_bar_hidden = True
        self.page.window.title_bar_buttons_hidden = True
        
        self.page.window.always_on_top = True
        self.page.window.opacity = 0.3
        self.page.window.bgcolor = ft.Colors.TRANSPARENT
        self.page.bgcolor = ft.Colors.with_opacity(0.3, ft.Colors.BLACK) # 蟆代＠證励￥縺励※謫堺ｽ應ｸｭ縺ｧ縺ゅｋ縺薙→繧堤､ｺ縺・
        
        # 繧ｪ繝ｼ繝舌・繝ｬ繧､繧定｡ｨ遉ｺ
        self.overlay_stack.visible = True
        self.page.update()

    def _on_pan_start(self, e: ft.DragStartEvent):
        """繝峨Λ繝・げ髢句ｧ具ｼ夐幕蟋狗せ繧定ｨ倬鹸"""
        self.start_x = e.local_x
        self.start_y = e.local_y
        self.selection_box.visible = True
        self.selection_box.left = self.start_x
        self.selection_box.top = self.start_y
        self.selection_box.width = 0
        self.selection_box.height = 0
        self.page.update()

    def _on_pan_update(self, e: ft.DragUpdateEvent):
        """繝峨Λ繝・げ荳ｭ・夐∈謚樒洸蠖｢繧呈緒逕ｻ"""
        self.current_x = e.local_x
        self.current_y = e.local_y

        # 蟾ｦ荳雁ｺｧ讓吶→蟷・・鬮倥＆繧定ｨ育ｮ・
        left = min(self.start_x, self.current_x)
        top = min(self.start_y, self.current_y)
        width = abs(self.current_x - self.start_x)
        height = abs(self.current_y - self.start_y)

        self.selection_box.left = left
        self.selection_box.top = top
        self.selection_box.width = width
        self.selection_box.height = height
        self.page.update()

    def _on_pan_end(self, e: ft.DragEndEvent):
        """Finish drag and capture selected region."""
        # 1. 縺ｾ縺哥let荳翫・隲也炊蠎ｧ讓・Logic Coordinates)繧定ｨ育ｮ・
        x1_local = min(self.start_x, self.current_x)
        y1_local = min(self.start_y, self.current_y)
        x2_local = max(self.start_x, self.current_x)
        y2_local = max(self.start_y, self.current_y)

        # --- 菫ｮ豁｣: 繧ｦ繧｣繝ｳ繝峨え縺ｮ邨ｶ蟇ｾ菴咲ｽｮ・医が繝輔そ繝・ヨ・峨ｒ蜿門ｾ励＠縺ｦ蜉邂・---
        # Flet縺ｮ繧ｦ繧｣繝ｳ繝峨え縺檎判髱｢縺ｮ縺ｩ縺薙↓縺ゅｋ縺具ｼ医Γ繝九Η繝ｼ繝舌・蛻・★繧後※縺・ｋ縺具ｼ峨ｒ蜿門ｾ・
        # 蛟､縺・None 縺ｮ蝣ｴ蜷医・ 0 縺ｨ縺吶ｋ
        offset_x = self.page.window.left or 0
        offset_y = self.page.window.top or 0
        
        # 繧ｦ繧｣繝ｳ繝峨え菴咲ｽｮ + 繧ｳ繝ｳ繝・リ蜀・・菴咲ｽｮ = 逕ｻ髱｢蜈ｨ菴薙・邨ｶ蟇ｾ蠎ｧ讓・
        x1_global = x1_local + offset_x
        y1_global = y1_local + offset_y
        x2_global = x2_local + offset_x
        y2_global = y2_local + offset_y
        # -------------------------------------------------------

        # 2. 繧ｹ繧ｱ繝ｼ繝ｫ繝輔ぃ繧ｯ繧ｿ繝ｼ繧呈寺縺代※迚ｩ逅・ｺｧ讓・Physical Coordinates)縺ｫ螟画鋤
        self.x1_phys = int(x1_global * self.scale_factor)
        self.y1_phys = int(y1_global * self.scale_factor)
        self.x2_phys = int(x2_global * self.scale_factor)
        self.y2_phys = int(y2_global * self.scale_factor)
        
        self._capture_and_restore(self.x1_phys, self.y1_phys, self.x2_phys, self.y2_phys)

    def _capture_and_restore(self, x1, y1, x2, y2):
        """Capture selected screen area and restore window."""
        
        # 1. 閾ｪ霄ｫ縺ｮ繧ｦ繧｣繝ｳ繝峨え縺悟・繧願ｾｼ縺ｾ縺ｪ縺・ｈ縺・↓螳悟・縺ｫ髫縺・
        self.page.window.opacity = 0
        self.page.update()
        
        # 繧ｦ繧｣繝ｳ繝峨え縺梧ｶ医∴繧九い繝九Γ繝ｼ繧ｷ繝ｧ繝ｳ遲峨ｒ蠕・▽縺溘ａ縺ｮ蠕ｮ蟆上↑蠕・ｩ・
        time.sleep(0.2)

        # 2. 繧ｹ繧ｯ繝ｪ繝ｼ繝ｳ繧ｷ繝ｧ繝・ヨ蜿門ｾ・(PIL)
        # width/height縺悟ｰ上＆縺吶℃繧句ｴ蜷医・辟｡隕・
        if (x2 - x1) > 5 and (y2 - y1) > 5:
            try:
                self.im_crop = ImageGrab.grab(bbox=(x1, y1, x2, y2)).convert("RGB")
                self.cboc_fixed.disabled=False
                # 逕ｻ蜒上ｒBase64譁・ｭ怜・縺ｫ螟画鋤・・let縺ｧ陦ｨ遉ｺ縺吶ｋ縺溘ａ・・
                buffered = io.BytesIO()
                self.im_crop.save(buffered, format="png")
                self.img_control.src_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                self.img_control.src = None
                self.result_crop_image.src_base64=self.img_control.src_base64
                self.page.open(self.dialog)
            except Exception as ex:
                print(f"Capture failed: {ex}")

        # 3. 繧ｦ繧｣繝ｳ繝峨え險ｭ螳壹ｒ蠕ｩ蜈・
        self.overlay_stack.visible = False
        self.selection_box.visible = False
        
        self.page.window.opacity = 1
        self.page.window.maximized = False
        self.page.window.title_bar_hidden = False
        self.page.window.title_bar_buttons_hidden = False
        #self.page.window.frameless = False
        self.page.window.always_on_top = False
        self.page.window.bgcolor = ft.Colors.WHITE # 蜈・・濶ｲ縺ｫ謌ｻ縺・
        self.page.bgcolor = self.original_bgcolor
        self.page.update()
        time.sleep(0.2)

        # 菴咲ｽｮ縺ｨ繧ｵ繧､繧ｺ繧呈綾縺・
        self.page.window.width = self.original_width
        self.page.window.height = self.original_height
        self.page.window.left = self.original_left
        self.page.window.top = self.original_top
        
        self.page.update()

    def mini_ocr(self,e):
        if self.im_crop is None:
            return
        # OCR荳ｭ縺ｮ繝懊ち繝ｳ辟｡蜉ｹ蛹・
        self.ocr_btn.disabled = True
        self.resultsmessage.value=""
        self.page.update()
        try:
            allstart=time.time()
            filename_base = "captureimg"
            pdf_tmp_path = getattr(globals(), "PDFTMPPATH", "tmp") # 譛ｪ螳夂ｾｩ蟇ｾ遲・
            self.outputcroppedpath = os.path.join(self.outputdirpath, pdf_tmp_path, f"{filename_base}.jpg")
            tatelinecnt = 0
            alllinecnt = 0

            npimg = np.array(self.im_crop)
            img_h, img_w = npimg.shape[:2]
            
            detections, classeslist = ocr.process_detector(
                detector=self.detector,
                inputname=filename_base,
                npimage=npimg,
                outputpath=self.outputdirpath,
                issaveimg=False
            )

            resultobj = [dict(), dict()]
            resultobj[0][0] = list()
            for i in range(17):
                resultobj[1][i] = []
                
            for det in detections:
                xmin, ymin, xmax, ymax = det["box"]
                conf = det["confidence"]
                if det["class_index"] == 0:
                    resultobj[0][0].append([xmin, ymin, xmax, ymax])
                resultobj[1][det["class_index"]].append([xmin, ymin, xmax, ymax, conf])

            xmlstr = convert_to_xml_string3(
                img_w, img_h, filename_base, classeslist, resultobj)
            xmlstr = "<OCRDATASET>" + xmlstr + "</OCRDATASET>"
            root = ET.fromstring(xmlstr)
            
            eval_xml(root, logger=None)

            alllineobj = []
            alltextlist = []

            for idx, lineobj in enumerate(root.findall(".//LINE")):
                xmin = int(lineobj.get("X"))
                ymin = int(lineobj.get("Y"))
                line_w = int(lineobj.get("WIDTH"))
                line_h = int(lineobj.get("HEIGHT"))
                try:
                    pred_char_cnt = float(lineobj.get("PRED_CHAR_CNT"))
                except:
                    pred_char_cnt = 100.0
                
                if line_h > line_w:
                    tatelinecnt += 1
                alllinecnt += 1

                # 驛ｨ蛻・判蜒上・蛻・ｊ蜃ｺ縺・
                lineimg = npimg[ymin:ymin+line_h, xmin:xmin+line_w, :]
                linerecogobj = RecogLine(lineimg, idx, pred_char_cnt)
                
                alllineobj.append(linerecogobj)

            # 隱崎ｭ倥・繝ｭ繧ｻ繧ｹ
            resultlinesall = process_cascade(
                alllineobj, self.recognizer30, self.recognizer50, self.recognizer100, is_cascade=True
            )
            resultlinesall = list(resultlinesall)
            alltextlist.append("\n".join(resultlinesall))
            resjsonarray=[]
            for idx,lineobj in enumerate(root.findall(".//LINE")):
                lineobj.set("STRING",resultlinesall[idx])
                xmin=int(lineobj.get("X"))
                ymin=int(lineobj.get("Y"))
                line_w=int(lineobj.get("WIDTH"))
                line_h=int(lineobj.get("HEIGHT"))
                try:
                    conf=float(lineobj.get("CONF"))
                except:
                    conf=0
                jsonobj={"boundingBox": [[xmin,ymin],[xmin,ymin+line_h],[xmin+line_w,ymin],[xmin+line_w,ymin+line_h]],
                    "id": idx,"isVertical": "true","text": resultlinesall[idx],"isTextline": "true","confidence": conf}
                resjsonarray.append(jsonobj)
            receipt_fields = extract_receipt_fields(resultlinesall, line_items=resjsonarray)
            receipt_summary = format_receipt_fields(receipt_fields, self.langcode)
            # 邵ｦ譖ｸ縺阪・讓ｪ譖ｸ縺榊愛螳壹Ο繧ｸ繝・け・亥盾閠・ｮ溯｣・∪縺ｾ・・
            if alllinecnt == 0 or tatelinecnt/alllinecnt > 0.5:
                alltextlist = alltextlist[::-1] # 騾・・↓縺吶ｋ

            # 邨先棡繝・く繧ｹ繝医・邨仙粋
            final_text = "\n".join(alltextlist)
            # UI縺ｸ縺ｮ蜿肴丐
            self.resultsmessage.value="{:.2f} sec".format(time.time()-allstart)
            self.resulttext.value = receipt_summary + "\n\n" + final_text
            self.result_jsonstr=json.dumps({"lines":resjsonarray,"receipt_fields":receipt_fields},ensure_ascii=False)
            self.open_result_page()

        except Exception as e:
            print(f"OCR Error: {e}")
            self.resulttext.value = f"繧ｨ繝ｩ繝ｼ縺檎匱逕溘＠縺ｾ縺励◆: {e}"
            self.open_result_page()
        finally:
            self.ocr_btn.disabled = False
            self.page.update()
    def open_dialog(self, e=None):
        self.start_capture()
        self.page.overlay.append(self.dialog)
        self.dialog.open = True
        self.page.update()

    def close_dialog(self, e):
        self.dialog.open = False
        self.page.update()

    def open_result_page(self):
        self.dialog.open = False
        self.page.overlay.append(self.result_dialog)
        self.result_dialog.open = True
        self.page.update()

    def close_result_page(self, e):
        self.result_dialog.open = False
        self.dialog.open = True
        self.page.update()

    def open_bibdlg_page(self,content):
        self.result_dialog.open = False
        self.bibinfo_dialog.open = True
        self.page.update()

    def close_bibinfo_page(self, e):
        self.bibinfo_dialog.open=False
        self.result_dialog.open = True
        self.page.update()
    
    def save_config(self):
        with open('userconf.yaml','w',encoding='utf-8')as wf:
            yaml.dump(self.config_obj, wf, default_flow_style=False, allow_unicode=True)


def main(page: ft.Page):
    parser = argparse.ArgumentParser(description="Argument for Inference using ONNXRuntime")
    parser.add_argument("--det-weights", type=str, required=False, help="Path to rtmdet onnx file", default="./src/model/deim-s-1024x1024.onnx")
    parser.add_argument("--det-classes", type=str, required=False, help="Path to list of class in yaml file",default="./src/config/ndl.yaml")
    parser.add_argument("--det-score-threshold", type=float, required=False, default=0.2)
    parser.add_argument("--det-conf-threshold", type=float, required=False, default=0.25)
    parser.add_argument("--det-iou-threshold", type=float, required=False, default=0.2)

    parser.add_argument("--rec-weights30", type=str, required=False, help="Path to parseq-tiny onnx file", default="./src/model/parseq-ndl-16x256-30-tiny-192epoch-tegaki3.onnx")
    parser.add_argument("--rec-weights50", type=str, required=False, help="Path to parseq-tiny onnx file", default="./src/model/parseq-ndl-16x384-50-tiny-146epoch-tegaki2.onnx")
    parser.add_argument("--rec-weights", type=str, required=False, help="Path to parseq-tiny onnx file", default="./src/model/parseq-ndl-16x768-100-tiny-165epoch-tegaki2.onnx")
    parser.add_argument("--rec-classes", type=str, required=False, help="Path to list of class in yaml file", default="./src/config/NDLmoji.yaml")
    parser.add_argument("--device", type=str, required=False, help="Device use (cpu or cuda)", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    
    page.title = "NDLOCR-Lite-GUI"
    page.window.icon=os.path.join("assets","icon.png")
    page.window.width = 1600
    page.window.height = 900
    page.window.min_width = 1400
    page.window.min_height = 860
    page.window.icon=os.path.join("assets","icon.png")

    default_config ={"langcode":"ja",
                     "json":True,
                     "xml":True,
                     "tei":True,
                     "txt":True,
                     "pdf":False,
                     "pdf_viztxt":False,
                     "selected_output_path":None,
                     "prompt":""
                     }
    load_obj={}
    if os.path.exists("userconf.yaml"):
        with open('userconf.yaml', encoding='utf-8')as f:
            load_obj= yaml.safe_load(f)
        if load_obj is None:
            load_obj={}

    config_obj=default_config|load_obj

    page.locale_configuration = ft.LocaleConfiguration(
        supported_locales=[
            ft.Locale("ja", "JP"),
            ft.Locale("en", "US")
        ], 
        current_locale=ft.Locale("ja", "JP") if config_obj["langcode"]=="ja" else ft.Locale("en", "US")
    )
    def save_config():
        with open('userconf.yaml','w',encoding='utf-8')as wf:
            yaml.dump(config_obj, wf, default_flow_style=False, allow_unicode=True)
    
    def handle_locale_change(e):
        index = e.control.selected_index
        if index == 0:
            page.locale_configuration.current_locale = ft.Locale("ja", "JP")
        elif index == 1:
            page.locale_configuration.current_locale = ft.Locale("en", "US")
        config_obj["langcode"]=page.locale_configuration.current_locale.language_code
        save_config()
        page.update()
        renderui()
    #繝｢繝・Ν縺ｮ繝ｭ繝ｼ繝峨・驥阪◆縺・・縺ｧ逕ｻ髱｢譖ｴ譁ｰ縺ｨ縺ｯ迢ｬ遶九＠縺ｦ譛蛻・蝗槭□縺・
    origin_detector=ocr.get_detector(args=args)
    origin_recognizer=ocr.get_recognizer(args=args)
    origin_recognizer30=ocr.get_recognizer(args=args,weights_path=args.rec_weights30)
    origin_recognizer50=ocr.get_recognizer(args=args,weights_path=args.rec_weights50)

    def renderui():
        page.clean()
        page.update()
        inputpathlist=[]
        visualizepathlist=[]
        outputtxtlist=[]
        outputreceiptlist=[]
        outputhighlightlist=[]
        outputfocuslist=[]
        outputjsonpathlist=[]
        outputjsonobjlist=[]
        selected_focus_field=""
        preview_zoom=1.0
        base_preview_width=760

        def create_pdf_func(outputpath:str,img:object,bboxlistobj:dict,viztxtflag:bool):
            import reportlab
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import portrait
            from reportlab.pdfbase import pdfmetrics
            from reportlab.pdfbase.cidfonts import UnicodeCIDFont
            from reportlab.lib.units import mm
            from reportlab.lib.utils import ImageReader
            from reportlab.lib.colors import blue
            
            print((img.shape[1],img.shape[0]))
            c = canvas.Canvas(outputpath, pagesize=(img.shape[1],img.shape[0]))
            pdfmetrics.registerFont(UnicodeCIDFont('HeiseiMin-W3', isVertical=True))
            pdfmetrics.registerFont(UnicodeCIDFont('HeiseiKakuGo-W5', isVertical=False))
            pilimg_data = io.BytesIO()
            pilimg=Image.fromarray(img)
            pilimg.save(pilimg_data, format='png')
            pilimg_data.seek(0)
            side_out = ImageReader(pilimg_data)
            #Image.fromarray(new_image)
            c.drawImage(side_out,0,0)
            if viztxtflag:
                c.setFillColor(blue)
            else:
                c.setFillColor(blue,alpha=0.0)
            for bboxobj in bboxlistobj:
                bbox=bboxobj["boundingBox"]
                text=bboxobj["text"]
                if abs(bbox[2][0]-bbox[0][0])<abs(bbox[1][1]-bbox[0][1]):
                    x_center=(bbox[0][0]+bbox[2][0])//2
                    y_center=img.shape[0]-bbox[0][1]
                    c.setFont('HeiseiMin-W3', abs(bbox[2][0]-bbox[0][0])*3//4)
                    c.drawString(x_center,y_center, text)
                else:
                    
                    x_center=min(bbox[0][0],bbox[2][0])
                    y_center=img.shape[0]-(bbox[0][1]+bbox[1][1])//2
                    c.setFont('HeiseiKakuGo-W5', abs(bbox[1][1]-bbox[0][1]))
                    c.drawString(x_center,y_center, text)
            c.save()
        

        def parts_control(flag:bool):
            file_upload_btn.disabled=flag
            directory_upload_btn.disabled=flag
            directory_output_btn.disabled=flag
            chkbx_visualize.disabled=flag
            customize_btn.disabled=flag
            preview_prev_btn.disabled=flag
            preview_next_btn.disabled=flag
            ocr_btn.disabled=flag
            crop_btn.disabled=flag
            cap_btn.disabled=flag
            localebutton.disabled=flag

        def get_base_preview_src(index: int) -> str:
            if len(visualizepathlist)>0 and 0<=index<len(visualizepathlist):
                return visualizepathlist[index]
            if 0<=index<len(inputpathlist):
                return inputpathlist[index]
            return "dummy.dat"

        def bbox_to_text(bbox: List[int]) -> str:
            if not bbox or len(bbox) != 4:
                return "-"
            return f"({bbox[0]}, {bbox[1]})-({bbox[2]}, {bbox[3]})"

        def update_focus_buttons(index: int):
            if not (0<=index<len(outputreceiptlist)):
                store_btn.text = "店舗名: -"
                datetime_btn.text = "日時: -"
                total_btn.text = "合計: -"
                bbox_info.value = "座標: -"
                editor_store.value = ""
                editor_datetime.value = ""
                editor_total.value = ""
                return
            f = outputreceiptlist[index]
            store_btn.text = f"店舗名: {f.get('store_name') or '未検出'}"
            datetime_btn.text = f"日時: {f.get('datetime') or '未検出'}"
            total_btn.text = f"合計: {f.get('total') or '未検出'}"
            editor_store.value = f.get("store_name", "") or ""
            editor_datetime.value = f.get("datetime", "") or ""
            editor_total.value = f.get("total", "") or ""
            if selected_focus_field == "store_name":
                bbox_info.value = f"座標(store): {bbox_to_text(f.get('store_name_bbox', []))}"
            elif selected_focus_field == "datetime":
                bbox_info.value = f"座標(datetime): {bbox_to_text(f.get('datetime_bbox', []))}"
            elif selected_focus_field == "total":
                bbox_info.value = f"座標(total): {bbox_to_text(f.get('total_bbox', []))} / label: {bbox_to_text(f.get('total_label_bbox', []))}"
            else:
                bbox_info.value = "座標: -"

        def save_current_receipt_fields(e=None):
            nonlocal preview_index
            if not (0 <= preview_index < len(outputreceiptlist)):
                return
            current = outputreceiptlist[preview_index]
            updated = {
                "store_name": (editor_store.value or "").strip(),
                "datetime": (editor_datetime.value or "").strip(),
                "total": (editor_total.value or "").strip(),
                "store_name_bbox": current.get("store_name_bbox", []),
                "datetime_bbox": current.get("datetime_bbox", []),
                "total_bbox": current.get("total_bbox", []),
                "total_label_bbox": current.get("total_label_bbox", []),
            }
            outputreceiptlist[preview_index] = updated
            if 0 <= preview_index < len(outputjsonobjlist):
                outputjsonobjlist[preview_index]["receipt_fields"] = updated
            if 0 <= preview_index < len(outputjsonpathlist):
                try:
                    with open(outputjsonpathlist[preview_index], "w", encoding="utf-8") as wf:
                        wf.write(json.dumps(outputjsonobjlist[preview_index], ensure_ascii=False, indent=2))
                except Exception as ex:
                    progressmessage.value = f"保存失敗: {ex}"
                    progressmessage.update()
                    return
            update_focus_buttons(preview_index)
            store_btn.update()
            datetime_btn.update()
            total_btn.update()
            editor_store.update()
            editor_datetime.update()
            editor_total.update()
            bbox_info.update()
            progressmessage.value = "抽出項目を保存しました"
            progressmessage.update()

        def update_preview_image(index: int):
            base_src = get_base_preview_src(index)
            if selected_focus_field and 0<=index<len(outputhighlightlist):
                focus_src = ""
                if 0<=index<len(outputfocuslist):
                    focus_src = outputfocuslist[index].get(selected_focus_field, "")
                if not focus_src:
                    focus_src = outputhighlightlist[index].get(selected_focus_field)
                if focus_src and os.path.exists(focus_src):
                    preview_image.src = focus_src
                else:
                    preview_image.src = base_src
            else:
                preview_image.src = base_src
            preview_image.update()

        def set_focus_field(field_name: str):
            nonlocal selected_focus_field, preview_index
            selected_focus_field = field_name
            update_preview_image(preview_index)
            update_focus_buttons(preview_index)
            store_btn.update()
            datetime_btn.update()
            total_btn.update()
            bbox_info.update()

        def clear_focus_field(e=None):
            nonlocal selected_focus_field, preview_index
            selected_focus_field = ""
            update_preview_image(preview_index)
            update_focus_buttons(preview_index)
            store_btn.update()
            datetime_btn.update()
            total_btn.update()
            bbox_info.update()

        def zoom_preview(factor: float):
            nonlocal preview_zoom
            preview_zoom = max(0.3, min(6.0, preview_zoom * factor))
            preview_image.width = int(base_preview_width * preview_zoom)
            preview_image.update()

        def zoom_in_preview(e=None):
            zoom_preview(1.2)

        def zoom_out_preview(e=None):
            zoom_preview(1 / 1.2)

        def zoom_reset_preview(e=None):
            nonlocal preview_zoom
            preview_zoom = 1.0
            preview_image.width = base_preview_width
            preview_image.update()

        def _normalize_receipt_obj(fields: Dict) -> Dict:
            return {
                "store_name": (fields or {}).get("store_name", "") or "",
                "datetime": (fields or {}).get("datetime", "") or "",
                "total": (fields or {}).get("total", "") or "",
                "store_name_bbox": (fields or {}).get("store_name_bbox", []) or [],
                "datetime_bbox": (fields or {}).get("datetime_bbox", []) or [],
                "total_bbox": (fields or {}).get("total_bbox", []) or [],
                "total_label_bbox": (fields or {}).get("total_label_bbox", []) or [],
            }

        def _expected_viz_path(inputpath: str, outputpath: str) -> str:
            p = os.path.join(outputpath, "viz_" + os.path.basename(inputpath))
            if p.lower().endswith(".jp2"):
                p = p[:-4] + ".jpg"
            return p

        def load_existing_results_to_preview():
            nonlocal preview_index
            nonlocal outputtxtlist, outputreceiptlist, outputhighlightlist, outputfocuslist, outputjsonpathlist, outputjsonobjlist, visualizepathlist
            outdir = selected_output_path.value
            if not outdir or len(inputpathlist) == 0:
                return

            outputtxtlist.clear()
            outputreceiptlist.clear()
            outputhighlightlist.clear()
            outputfocuslist.clear()
            outputjsonpathlist.clear()
            outputjsonobjlist.clear()
            visualizepathlist.clear()
            loaded_any = False

            for inputpath in inputpathlist:
                stem = os.path.splitext(os.path.basename(inputpath))[0]
                txtpath = os.path.join(outdir, stem + ".txt")
                jsonpath = os.path.join(outdir, stem + ".json")
                receipt_jsonpath = os.path.join(outdir, stem + "_receipt_fields.json")
                json_use_path = jsonpath if os.path.exists(jsonpath) else receipt_jsonpath
                vizpath = _expected_viz_path(inputpath, outdir)
                visualizepathlist.append(vizpath if os.path.exists(vizpath) else inputpath)

                plain_text = ""
                if os.path.exists(txtpath):
                    try:
                        with open(txtpath, encoding="utf-8") as rf:
                            plain_text = rf.read()
                    except Exception:
                        plain_text = ""

                json_obj = None
                if os.path.exists(json_use_path):
                    try:
                        with open(json_use_path, encoding="utf-8") as jf:
                            json_obj = json.load(jf)
                    except Exception:
                        json_obj = None

                line_items = []
                if isinstance(json_obj, dict):
                    contents = json_obj.get("contents", [])
                    if isinstance(contents, list) and len(contents) > 0 and isinstance(contents[0], list):
                        line_items = contents[0]
                receipt_fields = _normalize_receipt_obj((json_obj or {}).get("receipt_fields", {}))
                if not (receipt_fields["store_name"] or receipt_fields["datetime"] or receipt_fields["total"]):
                    receipt_fields = _normalize_receipt_obj(extract_receipt_fields(plain_text.splitlines(), line_items=line_items))

                if not plain_text and line_items:
                    try:
                        plain_text = "\n".join([str(x.get("text", "")) for x in line_items if str(x.get("text", "")).strip()])
                    except Exception:
                        plain_text = ""

                if json_obj is None:
                    json_obj = {
                        "contents": [line_items],
                        "receipt_fields": receipt_fields,
                        "imginfo": {
                            "img_path": inputpath,
                            "img_name": os.path.basename(inputpath),
                        }
                    }
                else:
                    json_obj["receipt_fields"] = receipt_fields

                outputtxtlist.append(plain_text)
                outputreceiptlist.append(receipt_fields)
                outputjsonobjlist.append(json_obj)
                outputjsonpathlist.append(json_use_path if os.path.exists(json_use_path) else receipt_jsonpath)

                highlight_dir = os.path.join(outdir, "_preview_highlight")
                highlight_map = {}
                focus_map = {}
                p_store = os.path.join(highlight_dir, f"{stem}_store.jpg")
                if _draw_field_highlight(inputpath, p_store, [receipt_fields.get("store_name_bbox", [])]):
                    highlight_map["store_name"] = p_store
                p_store_focus = os.path.join(highlight_dir, f"{stem}_store_focus.jpg")
                if _draw_field_focus_view(inputpath, p_store_focus, receipt_fields.get("store_name_bbox", [])):
                    focus_map["store_name"] = p_store_focus
                p_datetime = os.path.join(highlight_dir, f"{stem}_datetime.jpg")
                if _draw_field_highlight(inputpath, p_datetime, [receipt_fields.get("datetime_bbox", [])]):
                    highlight_map["datetime"] = p_datetime
                p_datetime_focus = os.path.join(highlight_dir, f"{stem}_datetime_focus.jpg")
                if _draw_field_focus_view(inputpath, p_datetime_focus, receipt_fields.get("datetime_bbox", [])):
                    focus_map["datetime"] = p_datetime_focus
                p_total = os.path.join(highlight_dir, f"{stem}_total.jpg")
                if _draw_field_highlight(inputpath, p_total, [receipt_fields.get("total_label_bbox", []), receipt_fields.get("total_bbox", [])]):
                    highlight_map["total"] = p_total
                p_total_focus = os.path.join(highlight_dir, f"{stem}_total_focus.jpg")
                aux_boxes = [receipt_fields.get("total_label_bbox", [])] if receipt_fields.get("total_label_bbox", []) else []
                if _draw_field_focus_view(inputpath, p_total_focus, receipt_fields.get("total_bbox", []), aux_bboxes=aux_boxes):
                    focus_map["total"] = p_total_focus
                outputhighlightlist.append(highlight_map)
                outputfocuslist.append(focus_map)

                loaded_any = loaded_any or bool(plain_text) or os.path.exists(vizpath) or os.path.exists(json_use_path)

            if loaded_any and len(outputtxtlist) > 0:
                preview_index = 0
                preview_prev_btn.disabled = len(outputtxtlist) <= 1
                preview_next_btn.disabled = len(outputtxtlist) <= 1
                preview_text.value = outputtxtlist[0]
                current_visualizeimgname.value = os.path.basename(inputpathlist[0])
                update_preview_image(0)
                update_focus_buttons(0)
                preview_text.update()
                current_visualizeimgname.update()
                store_btn.update()
                datetime_btn.update()
                total_btn.update()
                editor_store.update()
                editor_datetime.update()
                editor_total.update()
                bbox_info.update()
                page.update()
            

        def ocr_button_result(e):
            progressbar.value=0
            outputpath=selected_output_path.value
            nonlocal inputpathlist,outputtxtlist,outputreceiptlist,outputhighlightlist,outputfocuslist,outputjsonpathlist,outputjsonobjlist,visualizepathlist,preview_index,args
            nonlocal selected_focus_field
            nonlocal origin_recognizer,origin_recognizer30,origin_recognizer50
            nonlocal origin_detector

            preview_index=0
            parts_control(True)
            page.update()
            progressmessage.value="Start"
            progressmessage.update()
            try:
                #detector=origin_detector
                tatelinecnt=0
                alllinecnt=0
                allsum=len(inputpathlist)
                allstart=time.time()
                progressbar.value=0
                progressbar.update()
                outputtxtlist.clear()
                outputreceiptlist.clear()
                outputhighlightlist.clear()
                outputfocuslist.clear()
                outputjsonpathlist.clear()
                outputjsonobjlist.clear()
                selected_focus_field=""
                update_focus_buttons(-1)
                store_btn.update()
                datetime_btn.update()
                total_btn.update()
                editor_store.update()
                editor_datetime.update()
                editor_total.update()
                bbox_info.update()
                visualizepathlist.clear()
                visualizepathlist=[]
                alljsonobjlist=[]
                for idx,inputpath in enumerate(inputpathlist):
                    progressmessage.value=inputpath
                    progressmessage.update()
                    pil_image = Image.open(inputpath).convert('RGB')
                    npimg = np.array(pil_image)
                    start = time.time()
                    inputdivlist=[]
                    imgnamelist=[]
                    inputdivlist.append(npimg)
                    imgnamelist.append(os.path.basename(inputpath))
                    allxmlstr="<OCRDATASET>\n"
                    alltextlist=[]
                    resjsonarray=[]
                    for img,imgname in zip(inputdivlist,imgnamelist):
                        img_h,img_w=img.shape[:2]
                        detections,classeslist=ocr.process_detector(detector=origin_detector,inputname=imgname,npimage=img,outputpath=outputpath,issaveimg=False)
                        e1=time.time()
                        print("layout detection Done!",e1-start)
                        #print(detections)
                        resultobj=[dict(),dict()]
                        resultobj[0][0]=list()
                        for i in range(17):
                            resultobj[1][i]=[]
                        for det in detections:
                            xmin,ymin,xmax,ymax=det["box"]
                            conf=det["confidence"]
                            char_count=det["pred_char_count"]
                            if det["class_index"]==0:
                                resultobj[0][0].append([xmin,ymin,xmax,ymax])
                            resultobj[1][det["class_index"]].append([xmin,ymin,xmax,ymax,conf,char_count])
                        #print(resultobj)
                        xmlstr=convert_to_xml_string3(img_w, img_h, imgname, classeslist, resultobj)
                        xmlstr="<OCRDATASET>"+xmlstr+"</OCRDATASET>"
                        #print(xmlstr)
                        root = ET.fromstring(xmlstr)
                        eval_xml(root, logger=None)
                        alllinerecogobj=[]
                        for idx,lineobj in enumerate(root.findall(".//LINE")):
                            xmin=int(lineobj.get("X"))
                            ymin=int(lineobj.get("Y"))
                            line_w=int(lineobj.get("WIDTH"))
                            line_h=int(lineobj.get("HEIGHT"))
                            try:
                                pred_char_cnt=float(lineobj.get("PRED_CHAR_CNT"))
                            except:
                                pred_char_cnt=0.0
                            if line_h>line_w:
                                tatelinecnt+=1
                            alllinecnt+=1
                            lineimg = img[ymin:ymin+line_h,xmin:xmin+line_w,:]
                            linerecogobj = RecogLine(lineimg,idx,pred_char_cnt)
                            alllinerecogobj.append(linerecogobj)
                        resultlinesall=process_cascade(alllinerecogobj,recognizer30=origin_recognizer30,recognizer50=origin_recognizer50,recognizer100=origin_recognizer)
                        alltextlist.append("\n".join(resultlinesall))
                        for idx,lineobj in enumerate(root.findall(".//LINE")):
                            lineobj.set("STRING",resultlinesall[idx])
                            xmin=int(lineobj.get("X"))
                            ymin=int(lineobj.get("Y"))
                            line_w=int(lineobj.get("WIDTH"))
                            line_h=int(lineobj.get("HEIGHT"))
                            try:
                                conf=float(lineobj.get("CONF"))
                            except:
                                conf=0
                            jsonobj={"boundingBox": [[xmin,ymin],[xmin,ymin+line_h],[xmin+line_w,ymin],[xmin+line_w,ymin+line_h]],
                                "id": idx,"isVertical": "true","text": resultlinesall[idx],"isTextline": "true","confidence": conf}
                            resjsonarray.append(jsonobj)
                        allxmlstr+=(ET.tostring(root.find("PAGE"), encoding='unicode')+"\n")
                        e2=time.time()
                    allxmlstr+="</OCRDATASET>"
                    if alllinecnt==0 or tatelinecnt/alllinecnt>0.5:
                        alltextlist=alltextlist[::-1]
                    plain_text = "\n".join(alltextlist)
                    receipt_fields = extract_receipt_fields(plain_text.splitlines(), line_items=resjsonarray)
                    outputtxtlist.append(plain_text)
                    outputreceiptlist.append(receipt_fields)
                    stem = os.path.splitext(os.path.basename(inputpath))[0]
                    highlight_dir = os.path.join(outputpath, "_preview_highlight")
                    highlight_map = {}
                    focus_map = {}
                    p_store = os.path.join(highlight_dir, f"{stem}_store.jpg")
                    if _draw_field_highlight(inputpath, p_store, [receipt_fields.get("store_name_bbox", [])]):
                        highlight_map["store_name"] = p_store
                    p_store_focus = os.path.join(highlight_dir, f"{stem}_store_focus.jpg")
                    if _draw_field_focus_view(inputpath, p_store_focus, receipt_fields.get("store_name_bbox", [])):
                        focus_map["store_name"] = p_store_focus
                    p_datetime = os.path.join(highlight_dir, f"{stem}_datetime.jpg")
                    if _draw_field_highlight(inputpath, p_datetime, [receipt_fields.get("datetime_bbox", [])]):
                        highlight_map["datetime"] = p_datetime
                    p_datetime_focus = os.path.join(highlight_dir, f"{stem}_datetime_focus.jpg")
                    if _draw_field_focus_view(inputpath, p_datetime_focus, receipt_fields.get("datetime_bbox", [])):
                        focus_map["datetime"] = p_datetime_focus
                    p_total = os.path.join(highlight_dir, f"{stem}_total.jpg")
                    if _draw_field_highlight(inputpath, p_total, [receipt_fields.get("total_label_bbox", []), receipt_fields.get("total_bbox", [])]):
                        highlight_map["total"] = p_total
                    p_total_focus = os.path.join(highlight_dir, f"{stem}_total_focus.jpg")
                    aux_boxes = [receipt_fields.get("total_label_bbox", [])] if receipt_fields.get("total_label_bbox", []) else []
                    if _draw_field_focus_view(inputpath, p_total_focus, receipt_fields.get("total_bbox", []), aux_bboxes=aux_boxes):
                        focus_map["total"] = p_total_focus
                    outputhighlightlist.append(highlight_map)
                    outputfocuslist.append(focus_map)
                    alljsonobj={
                        "contents":[resjsonarray],
                        "receipt_fields": receipt_fields,
                        "imginfo": {
                            "img_width": img_w,
                            "img_height": img_h,
                            "img_path":inputpath,
                            "img_name":os.path.basename(inputpath)
                        }
                    }
                    jsonoutpath = os.path.join(outputpath, os.path.basename(inputpath).split(".")[0] + (".json" if chkbx_json.value else "_receipt_fields.json"))
                    outputjsonpathlist.append(jsonoutpath)
                    outputjsonobjlist.append(alljsonobj)
                    alljsonobjlist.append(alljsonobj)
                    if chkbx_xml.value:
                        with open(os.path.join(outputpath,os.path.basename(inputpath).split(".")[0]+".xml"),"w",encoding="utf-8") as wf:
                            wf.write(allxmlstr)
                    if chkbx_visualize.value:
                        output_vizpath=os.path.join(outputpath,"viz_"+os.path.basename(inputpath))
                        if output_vizpath.split(".")[-1]=="jp2":
                            output_vizpath=output_vizpath[:-4]+".jpg"
                        visualizepathlist.append(output_vizpath)
                        origin_detector.drawxml_detections(npimg=img,xmlstr=allxmlstr,categories=categories_org_name_index,outputimgpath=output_vizpath)
                    if chkbx_json.value:
                        with open(os.path.join(outputpath,os.path.basename(inputpath).split(".")[0]+".json"),"w",encoding="utf-8") as wf:
                            wf.write(json.dumps(alljsonobj,ensure_ascii=False,indent=2))
                    else:
                        with open(os.path.join(outputpath,os.path.basename(inputpath).split(".")[0]+"_receipt_fields.json"),"w",encoding="utf-8") as wf:
                            wf.write(json.dumps(alljsonobj,ensure_ascii=False,indent=2))
                    if chkbx_txt.value:
                        with open(os.path.join(outputpath,os.path.basename(inputpath).split(".")[0]+".txt"),"w",encoding="utf-8") as wtf:
                            wtf.write(plain_text)
                    if chkbx_pdf.value:
                        create_pdf_func(os.path.join(outputpath,os.path.basename(inputpath).split(".")[0]+".pdf"),img,resjsonarray,chkbx_pdf_viztxt.value)
                        
                    progressbar.value+=1/allsum
                    preview_prev_btn.disabled=False
                    preview_next_btn.disabled=False
                    preview_text.value= outputtxtlist[preview_index]
                    current_visualizeimgname.value=os.path.basename(inputpathlist[preview_index])
                    update_preview_image(preview_index)
                    update_focus_buttons(preview_index)
                    store_btn.update()
                    datetime_btn.update()
                    total_btn.update()
                    bbox_info.update()
                    page.update()
                if config_obj["langcode"]=="ja":
                    progressmessage.value="{} 画像OCR完了 / 所要時間 {:.2f} 秒".format(allsum, time.time()-allstart)
                else:
                    progressmessage.value="{} images completed / Total time {:.2f} sec".format(allsum,time.time()-allstart)
                progressmessage.update()
                if chkbx_tei.value:
                    with open(os.path.join(outputpath,os.path.basename(inputpathlist[0]).split(".")[0]+"_tei.xml"),"wb") as wf:
                        allxmlstrtei=convert_tei(alljsonobjlist)
                        wf.write(allxmlstrtei)
            except Exception as e:
                print(e)
                progressmessage.value=e
                progressmessage.update()
            finally:
                parts_control(False)
                page.update()

        
        def pick_files_result(e: ft.FilePickerResultEvent):
            if e.files:
                selected_input_path.value=e.files[0].path
                nonlocal inputpathlist,outputtxtlist,outputreceiptlist,outputhighlightlist,outputfocuslist,selected_focus_field
                inputpathlist.clear()
                outputtxtlist.clear()
                outputreceiptlist.clear()
                outputhighlightlist.clear()
                outputfocuslist.clear()
                outputjsonpathlist.clear()
                outputjsonobjlist.clear()
                selected_focus_field=""
                ext=e.files[0].path.split(".")[-1]
                if ext=="pdf":
                    filestem=os.path.basename(e.files[0].path)[:-4]
                    if config_obj["langcode"]=="ja":
                        progressmessage.value="pdf繝輔ぃ繧､繝ｫ縺ｮ蜑榊・逅・ｸｭ窶ｦ窶ｦ {} ".format(e.files[0].path)
                    else:
                        progressmessage.value="preprocessing pdf窶ｦ窶ｦ {} ".format(e.files[0].path)
                    parts_control(True)
                    page.update()
                    for p in glob.glob(os.path.join(os.getcwd(),PDFTMPPATH,"*.jpg")):
                        if os.path.isfile(p):
                            os.remove(p)
                    os.makedirs(os.path.join(os.getcwd(),PDFTMPPATH), exist_ok=True)
                    doc = pypdfium2.PdfDocument(selected_input_path.value)
                    #pdfarray = doc.render(pypdfium2.PdfBitmap.to_pil,scale=100 / 72)
                    pdfarray=doc.render(pypdfium2.PdfBitmap.to_pil,
                                            page_indices = [i for i in range(len(doc))],
                                            scale = 100/72)
                    for ix,image in enumerate(list(pdfarray)):
                        outputtmppath=os.path.join(os.getcwd(),PDFTMPPATH,"{}_{:05}.jpg".format(filestem,ix))
                        inputpathlist.append(outputtmppath)
                        image=image.convert("RGB")
                        image.save(outputtmppath)
                    if config_obj["langcode"]=="ja":
                        progressmessage.value="PDFの前処理完了"
                    else:
                        progressmessage.value="Preprocessing of pdf complete"
                    parts_control(False)
                    ocr_btn.disabled=True
                    crop_btn.disabled=True
                    page.update()
                else:
                    inputpathlist.append(e.files[0].path)
                selector.set_image(inputpathlist)
                if selected_output_path.value!=None:
                    parts_control(False)
                    load_existing_results_to_preview()
            selected_input_path.update()
            page.update()

        def pick_directory_result(e: ft.FilePickerResultEvent):
            #print(e.path)
            if e.path:
                selected_input_path.value = e.path
                nonlocal inputpathlist,outputtxtlist,outputreceiptlist,outputhighlightlist,outputfocuslist,selected_focus_field
                inputpathlist.clear()
                outputtxtlist.clear()
                outputreceiptlist.clear()
                outputhighlightlist.clear()
                outputfocuslist.clear()
                outputjsonpathlist.clear()
                outputjsonobjlist.clear()
                selected_focus_field=""
                update_focus_buttons(-1)
                store_btn.update()
                datetime_btn.update()
                total_btn.update()
                bbox_info.update()
                cleanflag=False
                for inputname in os.listdir(e.path):
                    inputpath=os.path.join(e.path,inputname)
                    ext=inputpath.split(".")[-1]
                    if ext.lower() in ["jpg","png","tiff","jp2","tif","jpeg","bmp"] and os.path.isfile(inputpath):
                        inputpathlist.append(inputpath)
                    elif ext=="pdf" and os.path.isfile(inputpath):
                        filestem=os.path.basename(inputpath)[:-4]
                        if config_obj["langcode"]=="ja":
                            progressmessage.value="pdf繝輔ぃ繧､繝ｫ縺ｮ蜑榊・逅・ｸｭ窶ｦ窶ｦ {} ".format(e.files[0].path)
                        else:
                            progressmessage.value="preprocessing pdf窶ｦ窶ｦ {} ".format(e.files[0].path)
                        parts_control(True)
                        page.update()
                        if not cleanflag:
                            for p in glob.glob(os.path.join(os.getcwd(),PDFTMPPATH,"*.jpg")):
                                if os.path.isfile(p):
                                    os.remove(p)
                            cleanflag=True
                        os.makedirs(os.path.join(os.getcwd(),PDFTMPPATH), exist_ok=True)
                        doc = pypdfium2.PdfDocument(inputpath)
                        pdfarray=doc.render(pypdfium2.PdfBitmap.to_pil,
                                            page_indices = [i for i in range(len(doc))],
                                            scale = 100/72)
                        for ix,image in enumerate(list(pdfarray)):
                            outputtmppath=os.path.join(os.getcwd(),PDFTMPPATH,"{}_{:05}.jpg".format(filestem,ix))
                            inputpathlist.append(outputtmppath)
                            image=image.convert("RGB")
                            image.save(outputtmppath)

                        if config_obj["langcode"]=="ja":
                            progressmessage.value="PDFの前処理完了"
                        else:
                            progressmessage.value="Preprocessing of pdf complete"
                        parts_control(False)
                        crop_btn.disabled=True
                        ocr_btn.disabled=True
                        page.update()
                selector.set_image(inputpathlist)
                #print(inputpath)
            if selected_output_path.value!=None and len(inputpathlist)>0:
                parts_control(False)
                load_existing_results_to_preview()
            selected_input_path.update()
            page.update()

        def pick_output_result(e: ft.FilePickerResultEvent):
            nonlocal inputpathlist
            if e.path:
                selected_output_path.value = e.path
                selected_output_path.update()
                config_obj["selected_output_path"]=e.path
                save_config()
                selector.set_outputdir(e.path)
                if len(inputpathlist)>0:
                    parts_control(False)
                    load_existing_results_to_preview()
            page.update()

        preview_index=0
        def next_image(e):
            nonlocal inputpathlist,outputtxtlist,outputreceiptlist,preview_index
            if preview_index < min(len(inputpathlist) - 1,len(outputtxtlist) - 1):
                preview_index += 1
            else:
                preview_index = 0

            if 0<=preview_index<len(outputtxtlist):
                current_visualizeimgname.value=os.path.basename(inputpathlist[preview_index])
            if 0<=preview_index<len(outputtxtlist):
                preview_text.value=outputtxtlist[preview_index]
            update_preview_image(preview_index)
            update_focus_buttons(preview_index)
            preview_text.update()
            store_btn.update()
            datetime_btn.update()
            total_btn.update()
            bbox_info.update()
            page.update()


        def prev_image(e):
            nonlocal inputpathlist,outputtxtlist,outputreceiptlist,preview_index
            if preview_index > 0:
                preview_index -= 1
            else:
                preview_index = min(len(inputpathlist) - 1,len(outputtxtlist) - 1)
            
            if 0<=preview_index<len(outputtxtlist):
                current_visualizeimgname.value=os.path.basename(inputpathlist[preview_index])
            if 0<=preview_index<len(outputtxtlist):
                preview_text.value=outputtxtlist[preview_index]
            update_preview_image(preview_index)
            update_focus_buttons(preview_index)
            preview_text.update()
            store_btn.update()
            datetime_btn.update()
            total_btn.update()
            bbox_info.update()
            page.update()
        

        def handle_customize_dlg_modal_close(e):
            config_obj.update({
                "json":chkbx_json.value,
                "txt":chkbx_txt.value,
                "xml":chkbx_xml.value,
                "tei":chkbx_tei.value,
                "pdf":chkbx_pdf.value,
                "pdf_viztxt":chkbx_pdf_viztxt.value,
            })
            save_config()
            page.close(customize_dlg_modal)
        
        def change_pdfstatus(e):
            chkbx_pdf_viztxt.disabled=not chkbx_pdf.value
            chkbx_pdf_viztxt.update()
        

        preview_image=ft.Image(src="dummy.dat", width=base_preview_width, gapless_playback=True, fit=ft.ImageFit.FIT_WIDTH)
        preview_image_viewer = ft.InteractiveViewer(
            min_scale=0.3,
            max_scale=6.0,
            boundary_margin=ft.margin.all(30),
            content=preview_image,
        )
        store_btn = ft.TextButton(text="店舗名: -", on_click=lambda e: set_focus_field("store_name"))
        datetime_btn = ft.TextButton(text="日時: -", on_click=lambda e: set_focus_field("datetime"))
        total_btn = ft.TextButton(text="合計: -", on_click=lambda e: set_focus_field("total"))
        clear_focus_btn = ft.TextButton(text="ハイライト解除", on_click=clear_focus_field)
        editor_store = ft.TextField(label="店舗名（編集）", dense=True, expand=True)
        editor_datetime = ft.TextField(label="日時（編集）", dense=True, expand=True)
        editor_total = ft.TextField(label="合計（編集）", dense=True, expand=True)
        save_fields_btn = ft.ElevatedButton("抽出項目を保存", on_click=save_current_receipt_fields)
        bbox_info = ft.Text(value="座標: -", selectable=True)
        preview_text=ft.Text(value="",height=300,selectable=True)

        pick_directory_dialog = ft.FilePicker(on_result=pick_directory_result)
        pick_output_dialog = ft.FilePicker(on_result=pick_output_result)
        pick_files_dialog = ft.FilePicker(on_result=pick_files_result)
        progressbar = ft.ProgressBar(width=400,value=0)
        selected_input_path = ft.Text(selectable=True)
        selected_output_path = ft.Text(config_obj["selected_output_path"],selectable=True)
        current_visualizeimgname=ft.Text(selectable=True)
        progressmessage=ft.Text()
        chkbx_visualize = ft.Checkbox(label=TRANSLATIONS["main_visualize_label"][config_obj["langcode"]], value=True)
        chkbx_json = ft.Checkbox(label="JSON形式", value=config_obj["json"])
        chkbx_txt = ft.Checkbox(label="TXT形式", value=config_obj["txt"])
        chkbx_xml = ft.Checkbox(label="XML形式", value=config_obj["xml"])
        chkbx_tei = ft.Checkbox(label="TEI形式", value=config_obj["tei"])
        chkbx_pdf = ft.Checkbox(label="検索可能PDF", value=config_obj["pdf"], on_change=change_pdfstatus)
        chkbx_pdf_viztxt = ft.Checkbox(label="PDFに可視テキストを表示", value=config_obj["pdf_viztxt"], disabled=not chkbx_pdf.value)

        
        file_upload_btn=ft.ElevatedButton(
                        TRANSLATIONS["main_file_upload_btn"][config_obj["langcode"]],
                        icon=ft.Icons.UPLOAD_FILE,
                        on_click=lambda _: pick_files_dialog.pick_files(
                            allow_multiple=False
                        ),
                    )
        directory_upload_btn=ft.ElevatedButton(
                        TRANSLATIONS["main_directory_upload_btn"][config_obj["langcode"]],
                        icon=ft.Icons.FOLDER_OPEN,
                        on_click=lambda _: pick_directory_dialog.get_directory_path(),
                    )
        directory_output_btn=ft.ElevatedButton(
                        TRANSLATIONS["main_directory_output_btn"][config_obj["langcode"]],
                        on_click=lambda _: pick_output_dialog.get_directory_path(),
                    )
        ocr_btn=ft.ElevatedButton(text="OCR",
                                    on_click=ocr_button_result,
                                    style=ft.ButtonStyle(
                                        padding=30,
                                        shape=ft.RoundedRectangleBorder(radius=10)),
                                    disabled=True)
        preview_text_col = ft.Column(
            controls=[
                ft.Text("抽出項目 (クリックで位置表示)"),
                ft.Row([
                    ft.Column(
                        controls=[
                            ft.Container(content=store_btn, alignment=ft.alignment.center_left),
                            ft.Container(content=datetime_btn, alignment=ft.alignment.center_left),
                            ft.Container(content=total_btn, alignment=ft.alignment.center_left),
                            ft.Container(content=clear_focus_btn, alignment=ft.alignment.center_left),
                        ],
                        expand=1,
                        horizontal_alignment=ft.CrossAxisAlignment.START,
                        spacing=8,
                    ),
                    ft.Container(
                        content=ft.Column(
                            controls=[
                                editor_store,
                                editor_datetime,
                                editor_total,
                                ft.Row([save_fields_btn], alignment=ft.MainAxisAlignment.END),
                            ],
                            expand=1,
                            spacing=8,
                        ),
                        expand=1,
                        padding=ft.padding.only(right=18),
                    ),
                ], wrap=False, vertical_alignment=ft.CrossAxisAlignment.START),
                bbox_info,
                ft.Divider(),
                preview_text
            ],
            scroll=ft.ScrollMode.ALWAYS,
            expand=True
        )
        preview_prev_btn=ft.ElevatedButton(text=TRANSLATIONS["main_prev_btn"][config_obj["langcode"]], on_click=prev_image,disabled=True)
        preview_next_btn=ft.ElevatedButton(text=TRANSLATIONS["main_next_btn"][config_obj["langcode"]], on_click=next_image,disabled=True)
        customize_btn=ft.ElevatedButton(TRANSLATIONS["main_customize_btn"][config_obj["langcode"]], on_click=lambda e: page.open(customize_dlg_modal))
        customize_dlg_modal = ft.AlertDialog(
            modal=True,
            title=ft.Text(TRANSLATIONS["customize_dlg_modal_title"][config_obj["langcode"]]),
            content=ft.Text(TRANSLATIONS["customize_dlg_modal_explain"][config_obj["langcode"]]),
            actions=[
                chkbx_txt,
                chkbx_json,
                ft.Row([chkbx_xml,chkbx_tei]),
                ft.Row([chkbx_pdf,chkbx_pdf_viztxt]),
                ft.TextButton("OK", on_click=handle_customize_dlg_modal_close),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )
        selector = ImageSelector(page,config_obj,detector=origin_detector,
                                recognizer30=origin_recognizer30,
                                recognizer50=origin_recognizer50,
                                recognizer100=origin_recognizer,
                                outputdirpath=selected_output_path.value)

        capture_tool=CaptureTool(page,config_obj,detector=origin_detector,
                                recognizer30=origin_recognizer30,
                                recognizer50=origin_recognizer50,
                                recognizer100=origin_recognizer)
        page.overlay.extend([customize_dlg_modal,pick_files_dialog,pick_directory_dialog,pick_output_dialog,
                            selector.dialog,selector.zoom_dialog,selector.result_dialog,
                            capture_tool.dialog,capture_tool.result_dialog])
        crop_btn = ft.ElevatedButton(text="Crop&OCR",
                                        on_click=selector.open_dialog,
                                        style=ft.ButtonStyle(
                                            padding=10,
                                            shape=ft.RoundedRectangleBorder(radius=10)),
                                        disabled=True)
        cap_btn = ft.ElevatedButton(text=TRANSLATIONS["main_cap_btn"][config_obj["langcode"]],
                                        on_click=capture_tool.start_capture,
                                        style=ft.ButtonStyle(
                                            padding=10,
                                            shape=ft.RoundedRectangleBorder(radius=10)),
                                        disabled=False)
        explain_label=ft.Text(TRANSLATIONS["main_explain"][config_obj["langcode"]])
        localebutton=ft.CupertinoSlidingSegmentedButton(
                        selected_index=0 if config_obj["langcode"]=="ja" else 1,
                        thumb_color=ft.Colors.BLUE_400,
                        on_change=handle_locale_change,
                        controls=[ft.Text("日本語"), ft.Text("English")],
                    )
        left_panel = ft.Column(
            controls=[
                ft.Row([localebutton]),
                ft.Row([explain_label, cap_btn]),
                ft.Divider(),
                ft.Row([
                    file_upload_btn,
                    directory_upload_btn,
                    ft.Text(TRANSLATIONS["main_target_label"][config_obj["langcode"]]),
                    selected_input_path,
                ]),
                ft.Divider(),
                ft.Row([
                    directory_output_btn,
                    ft.Text(TRANSLATIONS["main_output_label"][config_obj["langcode"]]),
                    selected_output_path,
                ]),
                ft.Divider(),
                ft.Row([
                    ocr_btn,
                    crop_btn,
                    ft.Column([chkbx_visualize, customize_btn]),
                    ft.Column([progressmessage, progressbar]),
                ]),
                ft.Divider(),
                ft.Row([
                    ft.Text(TRANSLATIONS["main_preview_label"][config_obj["langcode"]]),
                    preview_prev_btn,
                    preview_next_btn,
                    current_visualizeimgname
                ]),
                preview_text_col,
            ],
            scroll=ft.ScrollMode.ALWAYS,
            expand=True
        )

        right_panel = ft.Column(
            controls=[
                ft.Text("表示画面"),
                ft.Row([
                    ft.TextButton("拡大 +", on_click=zoom_in_preview),
                    ft.TextButton("縮小 -", on_click=zoom_out_preview),
                    ft.TextButton("100%", on_click=zoom_reset_preview),
                ]),
                ft.Divider(),
                preview_image_viewer,
            ],
            scroll=ft.ScrollMode.ALWAYS,
            expand=True
        )

        page.add(
            ft.Row(
                controls=[
                    ft.Container(content=left_panel, expand=1, padding=10),
                    ft.VerticalDivider(width=1),
                    ft.Container(content=right_panel, expand=1, padding=10),
                ],
                expand=True,
                spacing=0,
            )
        )
        page.update()
    renderui()
ft.app(main,assets_dir="assets")
