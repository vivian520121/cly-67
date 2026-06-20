import os
import time
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

from .preprocessor import ImagePreprocessor, get_image_files
from .ocr_engine import OCREngine, SUPPORTED_LANGS
from .exporter import ResultExporter
from .statistics import ImageOCRResult, BatchStatistics

console = Console()


class BatchProcessor:
    def __init__(
        self,
        input_path: str,
        output_dir: str = "./ocr_output",
        lang: str = "ch",
        formats: list = None,
        preprocess_params: dict = None,
        use_gpu: bool = False,
        save_preview: bool = False,
    ):
        self.input_path = input_path
        self.output_dir = output_dir
        self.lang = lang
        self.formats = formats or ['txt', 'md']
        self.preprocess_params = preprocess_params or {}
        self.use_gpu = use_gpu
        self.save_preview = save_preview
        self.preprocessor = ImagePreprocessor(**self.preprocess_params)
        self.ocr_engine = None
        self.exporter = ResultExporter(output_dir, self.formats)
        self.statistics = BatchStatistics()
        self._preview_dir = os.path.join(output_dir, "_preprocessed")
        if self.save_preview:
            os.makedirs(self._preview_dir, exist_ok=True)

    def _get_ocr_engine(self):
        if self.ocr_engine is None:
            self.ocr_engine = OCREngine(lang=self.lang, use_gpu=self.use_gpu)
        return self.ocr_engine

    def switch_language(self, lang: str):
        if lang not in SUPPORTED_LANGS:
            raise ValueError(f"不支持的语言: {lang}")
        self.lang = lang
        if self.ocr_engine:
            self.ocr_engine.switch_language(lang)

    def _process_single_image(self, image_path: str) -> ImageOCRResult:
        image_name = os.path.basename(image_path)
        result = ImageOCRResult(
            image_path=image_path,
            image_name=image_name,
            language=SUPPORTED_LANGS.get(self.lang, self.lang),
        )
        try:
            t0 = time.time()
            processed_img, preprocess_meta = self.preprocessor.process(image_path)
            result.preprocess_time = time.time() - t0
            result.blur_score = preprocess_meta['blur_score']
            result.is_blurry = preprocess_meta['is_blurry']
            if self.save_preview:
                preview_path = os.path.join(self._preview_dir, f"preview_{image_name}")
                self.preprocessor.save_preview(processed_img, preview_path)
            t1 = time.time()
            ocr_engine = self._get_ocr_engine()
            ocr_results = ocr_engine.recognize(processed_img)
            result.ocr_time = time.time() - t1
            result.ocr_lines = ocr_results
            result.full_text = ocr_engine.get_full_text(ocr_results)
            result.char_count = len(result.full_text.replace('\n', ''))
            result.line_count = len(ocr_results)
            result.avg_confidence = ocr_engine.get_avg_confidence(ocr_results)
            export_meta = {
                'blur_score': result.blur_score,
                'is_blurry': result.is_blurry,
                'avg_confidence': result.avg_confidence,
                'language': result.language,
            }
            result.export_files = self.exporter.export(image_path, ocr_results, export_meta)
        except Exception as e:
            result.error = str(e)
        return result

    def _print_result_preview(self, result: ImageOCRResult):
        if result.error:
            console.print(f"  [red]✗ 错误: {result.error}[/red]")
            return
        status_icon = "[green]✓[/green]" if result.is_success else "[yellow]![/yellow]"
        blur_tag = " [red][模糊][/red]" if result.is_blurry else ""
        console.print(
            f"  {status_icon} {result.image_name} | "
            f"字符: [cyan]{result.char_count}[/cyan] | "
            f"行数: [cyan]{result.line_count}[/cyan] | "
            f"置信度: [magenta]{result.avg_confidence:.4f}[/magenta] | "
            f"用时: [blue]{result.preprocess_time + result.ocr_time:.2f}s[/blue]"
            f"{blur_tag}"
        )
        if result.full_text:
            preview = result.full_text[:100]
            if len(result.full_text) > 100:
                preview += "..."
            console.print(f"    [dim]内容预览: {preview}[/dim]")

    def process(self) -> BatchStatistics:
        image_files = get_image_files(self.input_path)
        if not image_files:
            console.print(f"[yellow]未找到支持的图片文件: {self.input_path}[/yellow]")
            return self.statistics
        console.print(f"\n[bold cyan]开始处理 {len(image_files)} 张图片...[/bold cyan]")
        console.print(f"[dim]语言: {SUPPORTED_LANGS.get(self.lang)} | "
                      f"输出格式: {', '.join(self.formats)} | "
                      f"输出目录: {self.output_dir}[/dim]\n")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=50),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[green]OCR 识别中...", total=len(image_files))
            for image_path in image_files:
                progress.update(task, description=f"[green]处理: {os.path.basename(image_path)}")
                result = self._process_single_image(image_path)
                self.statistics.add_result(result)
                self._print_result_preview(result)
                progress.advance(task)
        console.print("\n[bold green]处理完成！[/bold green]\n")
        self._print_summary()
        self.exporter.export_summary(self.statistics.to_summary_list())
        return self.statistics

    def _print_summary(self):
        stats = self.statistics
        table = Table(title="📊 批量识别统计", show_header=True, header_style="bold magenta")
        table.add_column("统计项", style="cyan", no_wrap=True)
        table.add_column("数值", style="yellow")
        table.add_row("处理图片总数", str(stats.total_images))
        table.add_row("识别成功数", f"[green]{stats.success_count}[/green]")
        table.add_row("识别失败数", f"[red]{stats.failed_count}[/red]")
        table.add_row("模糊图片数", f"[yellow]{stats.blurry_count}[/yellow]")
        table.add_row("总识别字符数", str(stats.total_chars))
        table.add_row("总识别行数", str(stats.total_lines))
        table.add_row("成功率", f"{stats.success_rate * 100:.1f}%")
        table.add_row("平均置信度", f"{stats.avg_confidence:.4f}")
        table.add_row("平均模糊度", f"{stats.avg_blur_score:.2f}")
        table.add_row("总耗时", f"{stats.total_time:.2f}s")
        console.print(Panel(table, border_style="green"))
        if stats.blurry_images:
            blurry_names = [r.image_name for r in stats.blurry_images]
            console.print(f"[yellow]⚠️  模糊图片列表: {', '.join(blurry_names)}[/yellow]")
        if stats.failed_images:
            failed_names = [r.image_name for r in stats.failed_images]
            console.print(f"[red]❌ 识别失败列表: {', '.join(failed_names)}[/red]")
        console.print(f"\n[dim]识别结果已保存至: {self.output_dir}/[/dim]")
