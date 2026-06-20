from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ImageOCRResult:
    image_path: str
    image_name: str
    ocr_lines: list = field(default_factory=list)
    full_text: str = ""
    char_count: int = 0
    line_count: int = 0
    avg_confidence: float = 0.0
    blur_score: float = 0.0
    is_blurry: bool = False
    preprocess_time: float = 0.0
    ocr_time: float = 0.0
    export_files: dict = field(default_factory=dict)
    error: Optional[str] = None
    language: str = "ch"

    @property
    def is_success(self) -> bool:
        return self.error is None and self.char_count > 0


@dataclass
class BatchStatistics:
    results: list = field(default_factory=list)

    def add_result(self, result: ImageOCRResult):
        self.results.append(result)

    @property
    def total_images(self) -> int:
        return len(self.results)

    @property
    def success_count(self) -> int:
        return sum(1 for r in self.results if r.is_success)

    @property
    def failed_count(self) -> int:
        return self.total_images - self.success_count

    @property
    def blurry_count(self) -> int:
        return sum(1 for r in self.results if r.is_blurry)

    @property
    def total_chars(self) -> int:
        return sum(r.char_count for r in self.results)

    @property
    def total_lines(self) -> int:
        return sum(r.line_count for r in self.results)

    @property
    def success_rate(self) -> float:
        if self.total_images == 0:
            return 0.0
        return self.success_count / self.total_images

    @property
    def avg_confidence(self) -> float:
        valid = [r for r in self.results if r.avg_confidence > 0]
        if not valid:
            return 0.0
        return sum(r.avg_confidence for r in valid) / len(valid)

    @property
    def avg_blur_score(self) -> float:
        return sum(r.blur_score for r in self.results) / self.total_images if self.total_images else 0.0

    @property
    def total_time(self) -> float:
        return sum(r.preprocess_time + r.ocr_time for r in self.results)

    @property
    def blurry_images(self) -> list:
        return [r for r in self.results if r.is_blurry]

    @property
    def failed_images(self) -> list:
        return [r for r in self.results if not r.is_success]

    def to_summary_list(self) -> list:
        summary = []
        for r in self.results:
            summary.append({
                'image_name': r.image_name,
                'image_path': r.image_path,
                'char_count': r.char_count,
                'line_count': r.line_count,
                'avg_confidence': r.avg_confidence,
                'blur_score': r.blur_score,
                'is_blurry': r.is_blurry,
                'is_success': r.is_success,
                'language': r.language,
            })
        return summary
