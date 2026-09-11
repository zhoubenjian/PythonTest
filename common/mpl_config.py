"""
matplotlib 中文显示全局配置
在任何需要绘图的脚本开头使用：
    from common.mpl_config import setup_chinese_font
    setup_chinese_font()
或者直接：
    import common.mpl_config
"""
import matplotlib.pyplot as plt


# 字体优先级列表，覆盖 Windows / macOS / Linux
CHINESE_FONTS = [
    # Windows 优先
    'SimHei', 'Microsoft YaHei',
    # macOS 优先
    'PingFang SC', 'Heiti TC',
    # Linux 优先
    'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans',
]


def setup_chinese_font():
    """配置 matplotlib 以正确显示中文和负号。"""
    plt.rcParams['font.sans-serif'] = CHINESE_FONTS
    plt.rcParams['axes.unicode_minus'] = False

# 模块被 import 时自动执行一次，方便直接 `import common.mpl_config`
setup_chinese_font()