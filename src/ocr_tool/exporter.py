import os
from datetime import datetime


class ResultExporter:
    def __init__(self, output_dir: str, formats: list = None):
        self.output_dir = output_dir
        self.formats = formats or ['txt', 'md']
        self._ensure_output_dir()

    def _ensure_output_dir(self):
        os.makedirs(self.output_dir, exist_ok=True)

    def _get_base_name(self, image_path: str) -> str:
        return os.path.splitext(os.path.basename(image_path))[0]

    def _get_output_path(self, image_path: str, fmt: str) -> str:
        base_name = self._get_base_name(image_path)
        return os.path.join(self.output_dir, f"{base_name}.{fmt}")

    def export_txt(self, image_path: str, full_text: str, metadata: dict) -> str:
        output_path = self._get_output_path(image_path, 'txt')
        lines = []
        lines.append(f"# 图片: {os.path.basename(image_path)}")
        lines.append(f"# 识别时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"# 模糊度得分: {metadata.get('blur_score', 0):.2f}")
        lines.append(f"# 平均置信度: {metadata.get('avg_confidence', 0):.4f}")
        lines.append("")
        lines.append("=" * 50)
        lines.append("")
        lines.append(full_text if full_text else "(未识别到文字)")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        return output_path

    def export_md(self, image_path: str, ocr_results: list, metadata: dict) -> str:
        output_path = self._get_output_path(image_path, 'md')
        lines = []
        lines.append(f"# OCR 识别结果 - {os.path.basename(image_path)}")
        lines.append("")
        lines.append("## 基本信息")
        lines.append("")
        lines.append(f"- **原始图片**: `{os.path.basename(image_path)}`")
        lines.append(f"- **识别时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"- **语言**: {metadata.get('language', '中文')}")
        lines.append(f"- **模糊度得分**: {metadata.get('blur_score', 0):.2f}")
        blur_status = "⚠️ 模糊图片" if metadata.get('is_blurry') else "✅ 清晰"
        lines.append(f"- **图片清晰度**: {blur_status}")
        lines.append(f"- **识别行数**: {len(ocr_results)}")
        char_count = sum(len(r['text']) for r in ocr_results)
        lines.append(f"- **识别字符数**: {char_count}")
        lines.append(f"- **平均置信度**: {metadata.get('avg_confidence', 0):.4f}")
        lines.append("")
        lines.append("## 识别内容")
        lines.append("")
        if ocr_results:
            lines.append("| 序号 | 识别文本 | 置信度 |")
            lines.append("|------|----------|--------|")
            for idx, result in enumerate(ocr_results, 1):
                text = result['text'].replace('|', '\\|')
                conf = f"{result['confidence']:.4f}"
                lines.append(f"| {idx} | {text} | {conf} |")
            lines.append("")
            lines.append("## 完整文本")
            lines.append("")
            lines.append("```")
            full_text = '\n'.join(r['text'] for r in ocr_results)
            lines.append(full_text if full_text else "(未识别到文字)")
            lines.append("```")
        else:
            lines.append("> ⚠️ 未识别到任何文字内容")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        return output_path

    def export(self, image_path: str, ocr_results: list, metadata: dict) -> dict:
        full_text = '\n'.join(r['text'] for r in ocr_results)
        exported = {}
        if 'txt' in self.formats:
            exported['txt'] = self.export_txt(image_path, full_text, metadata)
        if 'md' in self.formats:
            exported['md'] = self.export_md(image_path, ocr_results, metadata)
        return exported

    def export_summary(self, all_results: list) -> str:
        summary_path = os.path.join(self.output_dir, "_summary.md")
        lines = []
        lines.append("# OCR 批量识别汇总报告")
        lines.append("")
        lines.append(f"- **生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"- **处理图片总数**: {len(all_results)}")
        total_chars = sum(r.get('char_count', 0) for r in all_results)
        lines.append(f"- **总识别字符数**: {total_chars}")
        blurry_count = sum(1 for r in all_results if r.get('is_blurry'))
        lines.append(f"- **模糊图片数**: {blurry_count}")
        failed_count = sum(1 for r in all_results if r.get('char_count', 0) == 0)
        lines.append(f"- **识别失败数**: {failed_count}")
        lines.append("")
        lines.append("## 详细结果")
        lines.append("")
        lines.append("| 序号 | 图片 | 字符数 | 行数 | 清晰度 | 置信度 | 状态 |")
        lines.append("|------|------|--------|------|--------|--------|------|")
        for idx, result in enumerate(all_results, 1):
            status = "❌ 失败" if result.get('char_count', 0) == 0 else "✅ 成功"
            blur = "⚠️ 模糊" if result.get('is_blurry') else "✅ 清晰"
            lines.append(
                f"| {idx} | {result.get('image_name', '')} | "
                f"{result.get('char_count', 0)} | {result.get('line_count', 0)} | "
                f"{blur} | {result.get('avg_confidence', 0):.4f} | {status} |"
            )
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        return summary_path
