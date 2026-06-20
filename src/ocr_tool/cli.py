import os
import sys
import click
from rich.console import Console
from rich.panel import Panel
from rich import print as rprint

from .__init__ import __version__
from .preprocessor import SUPPORTED_FORMATS, ImagePreprocessor
from .ocr_engine import SUPPORTED_LANGS
from .batch_processor import BatchProcessor

console = Console()


def print_banner():
    banner = r"""
   ____   _____ _____    _______   ____  
  / __ \ / ____|  __ \  |__   __| / __ \ 
 | |  | | |    | |__) |____| |   | |  | |
 | |  | | |    |  _  /______| |   | |  | |
 | |__| | |____| | \ \      | |   | |__| |
  \____/ \_____|_|  \_\     |_|    \____/ 
                                          
    """
    console.print(f"[bold cyan]{banner}[/bold cyan]")
    console.print(f"[bold]离线图片OCR文字提取工具[/bold] - v{__version__}")
    console.print("[dim]基于 PaddleOCR 的本地离线识别，不上传任何图片[/dim]\n")


@click.group(invoke_without_command=True)
@click.version_option(version=__version__, prog_name='ocr-tool')
@click.pass_context
def cli(ctx):
    if ctx.invoked_subcommand is None:
        print_banner()
        click.echo(ctx.get_help())


@cli.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.option('-o', '--output', 'output_dir', type=click.Path(), default='./ocr_output',
              show_default=True, help='输出目录')
@click.option('-l', '--lang', type=click.Choice(list(SUPPORTED_LANGS.keys())),
              default='ch', show_default=True, help='识别语言 (ch=中文, en=英文)')
@click.option('-f', '--format', 'formats', multiple=True, type=click.Choice(['txt', 'md']),
              default=['txt', 'md'], show_default=True, help='导出格式 (可多选)')
@click.option('--contrast', type=float, default=1.5, show_default=True,
              help='对比度增强系数 (1.0=不增强, 1.2-2.0推荐)')
@click.option('--denoise', type=int, default=10, show_default=True,
              help='去噪强度 (0=不处理, 5-15推荐)')
@click.option('--sharpen/--no-sharpen', default=True, show_default=True,
              help='是否启用锐化')
@click.option('--threshold/--no-threshold', default=False, show_default=True,
              help='是否启用二值化 (适合低质量图片)')
@click.option('--deskew/--no-deskew', default=True, show_default=True,
              help='是否启用倾斜校正')
@click.option('--resize', type=int, default=2000, show_default=True,
              help='最大宽度限制 (像素)')
@click.option('--gpu/--cpu', default=False, show_default=True,
              help='是否使用GPU加速')
@click.option('--preview/--no-preview', default=False, show_default=True,
              help='是否保存预处理后的预览图片')
def batch(input_path, output_dir, lang, formats, contrast, denoise,
          sharpen, threshold, deskew, resize, gpu, preview):
    """批量识别图片中的文字 (支持文件或目录)"""
    print_banner()
    preprocess_params = {
        'contrast_alpha': contrast,
        'denoise_strength': denoise,
        'sharpen': sharpen,
        'threshold': threshold,
        'deskew': deskew,
        'resize_max_width': resize,
    }
    formats_list = list(formats) if isinstance(formats, tuple) else formats
    try:
        processor = BatchProcessor(
            input_path=input_path,
            output_dir=output_dir,
            lang=lang,
            formats=formats_list,
            preprocess_params=preprocess_params,
            use_gpu=gpu,
            save_preview=preview,
        )
        processor.process()
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  用户中断操作[/yellow]")
        sys.exit(1)
    except ImportError as e:
        console.print(f"[red]❌ 依赖缺失: {e}[/red]")
        console.print("[dim]请运行: pip install -r requirements.txt[/dim]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]❌ 运行错误: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.argument('image_path', type=click.Path(exists=True, dir_okay=False))
@click.option('-o', '--output', 'output_dir', type=click.Path(), default='./ocr_output',
              show_default=True, help='输出目录')
@click.option('-l', '--lang', type=click.Choice(list(SUPPORTED_LANGS.keys())),
              default='ch', show_default=True, help='识别语言')
@click.option('--contrast', type=float, default=1.5, show_default=True,
              help='对比度增强系数')
@click.option('--denoise', type=int, default=10, show_default=True, help='去噪强度')
@click.option('--sharpen/--no-sharpen', default=True, show_default=True)
@click.option('--threshold/--no-threshold', default=False, show_default=True)
@click.option('--deskew/--no-deskew', default=True, show_default=True)
@click.option('--gpu/--cpu', default=False, show_default=True)
def single(image_path, output_dir, lang, contrast, denoise, sharpen, threshold, deskew, gpu):
    """识别单张图片并彩色显示结果"""
    print_banner()
    preprocess_params = {
        'contrast_alpha': contrast,
        'denoise_strength': denoise,
        'sharpen': sharpen,
        'threshold': threshold,
        'deskew': deskew,
    }
    try:
        processor = BatchProcessor(
            input_path=image_path,
            output_dir=output_dir,
            lang=lang,
            formats=['txt', 'md'],
            preprocess_params=preprocess_params,
            use_gpu=gpu,
            save_preview=False,
        )
        stats = processor.process()
        if stats.results:
            result = stats.results[0]
            if result.full_text:
                console.print(Panel.fit(
                    f"[bold green]识别结果 ({result.image_name}):[/bold green]\n\n"
                    f"[bold cyan]{result.full_text}[/bold cyan]",
                    border_style="green",
                    title="📝 文字内容"
                ))
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  用户中断操作[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]❌ 运行错误: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.option('-o', '--output', 'output_dir', type=click.Path(), default='./ocr_output',
              show_default=True)
@click.option('--contrast', type=float, default=1.5, show_default=True)
@click.option('--denoise', type=int, default=10, show_default=True)
@click.option('--sharpen/--no-sharpen', default=True, show_default=True)
@click.option('--threshold/--no-threshold', default=False, show_default=True)
@click.option('--deskew/--no-deskew', default=True, show_default=True)
@click.option('--resize', type=int, default=2000, show_default=True)
def clean(input_path, output_dir, contrast, denoise, sharpen, threshold, deskew, resize):
    """仅执行图片预处理（不识别），用于测试参数效果"""
    print_banner()
    from .preprocessor import get_image_files
    image_files = get_image_files(input_path)
    if not image_files:
        console.print(f"[yellow]未找到图片文件: {input_path}[/yellow]")
        return
    clean_dir = os.path.join(output_dir, "_cleaned")
    os.makedirs(clean_dir, exist_ok=True)
    preprocessor = ImagePreprocessor(
        contrast_alpha=contrast,
        denoise_strength=denoise,
        sharpen=sharpen,
        threshold=threshold,
        deskew=deskew,
        resize_max_width=resize,
    )
    console.print(f"[bold cyan]开始预处理 {len(image_files)} 张图片...[/bold cyan]")
    from tqdm import tqdm
    for img_path in tqdm(image_files, desc="预处理", ncols=80):
        try:
            processed_img, meta = preprocessor.process(img_path)
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            out_path = os.path.join(clean_dir, f"{base_name}_cleaned.jpg")
            preprocessor.save_preview(processed_img, out_path)
            blur_status = "[red]模糊[/red]" if meta['is_blurry'] else "[green]清晰[/green]"
            console.print(
                f"  {os.path.basename(img_path)}: 模糊度={meta['blur_score']:.2f} "
                f"({blur_status}) -> {os.path.basename(out_path)}"
            )
        except Exception as e:
            console.print(f"  [red]✗ {os.path.basename(img_path)}: {e}[/red]")
    console.print(f"\n[bold green]预处理完成！结果保存在: {clean_dir}/[/bold green]")


@cli.command()
def langs():
    """查看支持的语言列表"""
    print_banner()
    console.print(Panel(
        "\n".join([f"  [bold cyan]{k}[/bold cyan] -> [yellow]{v}[/yellow]" for k, v in SUPPORTED_LANGS.items()]),
        title="🌐 支持的语言",
        border_style="cyan"
    ))


@cli.command()
def formats():
    """查看支持的图片格式"""
    print_banner()
    fmt_list = sorted(SUPPORTED_FORMATS)
    console.print(Panel(
        "  " + ", ".join([f"[bold green]{f}[/bold green]" for f in fmt_list]),
        title="📷 支持的图片格式",
        border_style="green"
    ))


if __name__ == '__main__':
    cli()
