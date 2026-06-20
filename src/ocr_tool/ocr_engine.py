import os
import numpy as np
from typing import Optional

SUPPORTED_LANGS = {
    'ch': '中文',
    'en': '英文',
}


class OCREngine:
    def __init__(self, lang: str = 'ch', use_gpu: bool = False, det: bool = True, rec: bool = True, cls: bool = False):
        if lang not in SUPPORTED_LANGS:
            raise ValueError(f"不支持的语言: {lang}，支持的语言有: {list(SUPPORTED_LANGS.keys())}")
        self.lang = lang
        self.use_gpu = use_gpu
        self.det = det
        self.rec = rec
        self.cls = cls
        self._engine = None
        self._init_engine()

    def _init_engine(self):
        try:
            from paddleocr import PaddleOCR
        except ImportError:
            raise ImportError(
                "PaddleOCR 未安装。请运行: pip install paddleocr paddlepaddle"
            )
        self._engine = PaddleOCR(
            use_angle_cls=self.cls,
            lang=self.lang,
            use_gpu=self.use_gpu,
            show_log=False,
            det_db_thresh=0.3,
            det_db_box_thresh=0.5,
            det_db_unclip_ratio=1.6,
            rec_batch_num=6,
        )

    def switch_language(self, lang: str):
        if lang not in SUPPORTED_LANGS:
            raise ValueError(f"不支持的语言: {lang}")
        if lang != self.lang:
            self.lang = lang
            self._init_engine()

    def recognize(self, image: np.ndarray) -> list[dict]:
        if self._engine is None:
            self._init_engine()
        result = self._engine.ocr(image, cls=self.cls)
        parsed_results = []
        if not result or not result[0]:
            return parsed_results
        for line in result[0]:
            if len(line) >= 2:
                bbox = line[0]
                text_info = line[1]
                if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                    text = text_info[0]
                    confidence = float(text_info[1])
                else:
                    text = str(text_info)
                    confidence = 1.0
                parsed_results.append({
                    'bbox': bbox,
                    'text': text,
                    'confidence': confidence,
                })
        return parsed_results

    def get_full_text(self, results: list[dict]) -> str:
        lines = [r['text'] for r in results]
        return '\n'.join(lines)

    def get_avg_confidence(self, results: list[dict]) -> float:
        if not results:
            return 0.0
        confidences = [r['confidence'] for r in results]
        return sum(confidences) / len(confidences)
