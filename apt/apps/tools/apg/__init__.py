"""
APG (APT Plugin Package) Tools

提供插件打包、解包和管理工具
"""

try:
    from apt.apps.tools.apg.packager import PluginPackager
except ImportError:
    PluginPackager = None

__all__ = ['PluginPackager']
