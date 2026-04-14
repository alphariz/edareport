from edareport.renderers.html import HtmlRenderer

try:
    from edareport.renderers.widget import WidgetRenderer

    __all__ = ["HtmlRenderer", "WidgetRenderer"]
except ImportError:
    __all__ = ["HtmlRenderer"]
