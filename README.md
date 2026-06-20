# OCR Tool - 离线图片OCR文字提取命令行工具

基于 **Python + Click + PaddleOCR** 开发的本地离线图片OCR工具，全程不上传任何图片，保护隐私安全。

## ✨ 功能特性

| 模块 | 功能说明 |
|------|----------|
| 📦 **Batch 批量识别** | 支持目录递归批量处理，识别 jpg/png/heic/bmp/tiff/webp 等格式 |
| 🌐 **Lang 语言切换** | 支持中文(`ch`)、英文(`en`)双语识别，一键切换 |
| 💾 **Export 导出** | 同步保存为 TXT 和 Markdown 文件，文件名与原图一一对应，自动生成汇总报告 |
| 🧹 **Clean 预处理** | 自动对比度增强(CLAHE)、非局部均值去噪、锐化、倾斜校正、二值化 |
| 📊 **Stat 统计** | 统计每张图片字符数/行数/置信度，自动标记模糊图片，识别失败高亮提示 |
| 🎨 **交互体验** | Rich 彩色终端输出 + TQDM 进度条 + 面板式结果展示 |

## 🚀 快速开始

### 1. 安装依赖

```bash
# 推荐使用 Python 3.8 - 3.11
pip install -r requirements.txt
```

> **注意**: PaddlePaddle 如需 GPU 版本请参考 [Paddle 官网](https://www.paddlepaddle.org.cn/) 安装对应 CUDA 版本。

### 2. 命令行使用

```bash
# 方式一：开发模式（直接运行）
python main.py --help

# 方式二：安装为命令
pip install -e .
ocr-tool --help
```

## 📖 命令详解

### 1. `batch` - 批量识别图片

```bash
# 基本用法：识别目录下所有图片（默认中文）
python main.py batch ./images

# 指定输出目录 + 英文识别
python main.py batch ./images -o ./result -l en

# 仅导出 Markdown 格式
python main.py batch ./images -f md

# 高质量手写识别推荐参数
python main.py batch ./handwritten \
  --contrast 2.0 \
  --denoise 15 \
  --sharpen \
  --threshold \
  --deskew
```

**所有参数**:
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `INPUT_PATH` | 路径 | 必填 | 图片文件或目录 |
| `-o, --output` | 路径 | `./ocr_output` | 输出目录 |
| `-l, --lang` | `ch`/`en` | `ch` | 识别语言 |
| `-f, --format` | `txt`/`md` | `txt,md` | 导出格式(可多选) |
| `--contrast` | float | `1.5` | 对比度系数(1.0-2.5) |
| `--denoise` | int | `10` | 去噪强度(0-20) |
| `--sharpen/--no-sharpen` | bool | `True` | 锐化开关 |
| `--threshold/--no-threshold` | bool | `False` | 二值化(低质量推荐开) |
| `--deskew/--no-deskew` | bool | `True` | 倾斜校正 |
| `--resize` | int | `2000` | 最大宽度限制(像素) |
| `--gpu/--cpu` | bool | `False` | GPU加速 |
| `--preview/--no-preview` | bool | `False` | 保存预处理预览图 |

### 2. `single` - 单张识别 + 彩色预览

```bash
python main.py single photo.jpg -l ch
```

识别完成后会在终端以彩色面板显示提取的文字内容。

### 3. `clean` - 仅预处理（不识别）

用于测试预处理参数效果，生成清洗后的图片：

```bash
python main.py clean ./images --contrast 1.8 --denoise 12
```

输出到 `./ocr_output/_cleaned/` 目录。

### 4. `langs` - 查看支持语言

```bash
python main.py langs
```

### 5. `formats` - 查看支持格式

```bash
python main.py formats
```

## 📂 输出目录结构

```
ocr_output/
├── image1.txt              # 纯文本识别结果
├── image1.md               # Markdown格式（含表格+元数据）
├── image2.txt
├── image2.md
├── _summary.md             # ✨ 批量识别汇总报告
└── _preprocessed/          # (可选) 预处理后的预览图
    └── preview_image1.jpg
```

## 🧪 参数调优指南

| 场景 | 推荐参数组合 |
|------|--------------|
| **普通印刷体** | 默认参数 `--contrast 1.5 --denoise 10` |
| **手写体/笔记** | `--contrast 2.0 --denoise 15 --sharpen --threshold` |
| **模糊/老旧图片** | `--contrast 1.8 --denoise 18 --sharpen --deskew` |
| **高质量扫描件** | `--contrast 1.2 --denoise 5 --no-threshold` |
| **照片/截图** | `--contrast 1.5 --denoise 10 --no-threshold` |

## 🔍 模糊检测说明

- 工具使用 **Laplacian 方差** 自动检测图片清晰度
- 方差 **< 100** 判定为模糊图片，结果中会标记 `[模糊]`
- 汇总报告中单独列出模糊图片，建议重拍或手动处理
- 模糊图片识别准确率会显著下降，可尝试增强预处理参数

## 📝 项目结构

```
cly-67/
├── main.py                     # 开发入口脚本
├── requirements.txt            # 依赖列表
├── pyproject.toml              # 项目配置
├── README.md                   # 使用说明
└── src/ocr_tool/
    ├── __init__.py
    ├── cli.py                  # Click CLI 主入口 (5个命令)
    ├── preprocessor.py         # 图像预处理模块
    ├── ocr_engine.py           # PaddleOCR 封装
    ├── batch_processor.py      # 批量处理 + Rich进度显示
    ├── exporter.py             # TXT/MD导出 + 汇总报告
    └── statistics.py           # 数据类 + 统计计算
```

## ⚠️ 常见问题

**Q: 首次运行很慢？**
A: PaddleOCR 首次运行会自动下载模型文件（~100MB），之后完全离线运行。

**Q: HEIC格式图片无法识别？**
A: 已内置 pillow-heif 支持，如仍失败请确保 macOS 系统已安装相关库。

**Q: 如何提高手写识别率？**
A: 1) 开启 `--threshold` 二值化  2) 提高 `--contrast` 到 1.8-2.5  3) 提高 `--denoise` 到 15+

**Q: 内存占用太高？**
A: 降低 `--resize` 到 1500 或 1000 可显著降低内存消耗。

## 📄 License

MIT License
