# APX + EQI 插件系统重构计划

**日期**: 2025-10-26
**任务**: 整合APX能力检测、EQI管理器与现有8个插件
**优先级**: 🔴 高

---

## 📋 重构目标

### 核心目标

1. ✅ **APX集成**: 将APX能力检测与插件系统深度集成
2. ✅ **EQI管理器**: 插件通过EQI系统进行智能激活
3. ✅ **版本检测**: 实现engine版本兼容性检查
4. ✅ **自动加载**: 基于模型能力自动加载插件
5. ✅ **插件适配**: 将8个现有插件适配新系统

### 建议操作实现

来自APX_COMPATIBILITY_REPORT.md的建议：

| 建议 | 优先级 | 状态 |
|------|--------|------|
| 修复apt_model.__init__.py导入 | P1 | ⏳ 待实现 |
| 添加自动插件加载 | P1 | ⏳ 待实现 |
| 扩展PluginManifest | P2 | ⏳ 待实现 |
| 创建APXLoader | P2 | ⏳ 待实现 |
| 添加版本检测 | P2 | ⏳ 待实现 |

---

## 🎯 阶段一：核心系统增强

### 任务1.1: 修复apt_model.__init__.py (P1)

**问题**：
```python
# 当前 apt_model/__init__.py
from apt_model.config.apt_config import APTConfig  # ← 需要torch
```

**解决方案**：延迟导入
```python
# 新的 apt_model/__init__.py
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from apt_model.config.apt_config import APTConfig

# 延迟导入函数
def get_config():
    """Lazy load config only when needed"""
    from apt_model.config.apt_config import APTConfig
    return APTConfig()

# tools子包不触发主包导入
__all__ = ['get_config']
```

**测试**：
```bash
# 应该成功（不需要torch）
python -c "from apt_model.tools.apx import pack_apx"
```

---

### 任务1.2: 扩展PluginManifest支持能力字段 (P2)

**当前**：
```python
@dataclass
class PluginManifest:
    name: str
    priority: int
    blocking: bool
    events: List[str]
    # ... 其他字段
```

**增强**：
```python
@dataclass
class PluginManifest:
    name: str
    priority: int
    blocking: bool
    events: List[str]
    # ... 现有字段

    # 新增：能力相关字段
    required_capabilities: List[str] = field(default_factory=list)
    optional_capabilities: List[str] = field(default_factory=list)
    provides_capabilities: List[str] = field(default_factory=list)

    # 新增：版本兼容性
    engine: str = ">=1.0.0"  # 最低引擎版本

    def matches_model(self, model_caps: List[str]) -> bool:
        """检查插件是否适用于模型"""
        if not self.required_capabilities:
            return True
        return all(cap in model_caps for cap in self.required_capabilities)

    def is_compatible_with_engine(self, engine_version: str) -> bool:
        """检查引擎版本兼容性"""
        # 实现semantic versioning比较
        return version_compatible(engine_version, self.engine)
```

---

### 任务1.3: 添加版本检测系统 (P2)

**新文件**：`apt_model/console/version_checker.py`

```python
from typing import Tuple
import re

def parse_version(version_str: str) -> Tuple[int, int, int]:
    """解析语义化版本号"""
    match = re.match(r'(\d+)\.(\d+)\.(\d+)', version_str)
    if not match:
        raise ValueError(f"Invalid version: {version_str}")
    return tuple(map(int, match.groups()))

def version_compatible(current: str, requirement: str) -> bool:
    """
    检查版本兼容性

    支持的格式：
    - ">=1.0.0"  - 大于等于
    - "~=1.2.0"  - 兼容版本 (1.2.x)
    - "==1.0.0"  - 精确匹配
    - "1.0.0"    - 默认为 >=
    """
    # 解析requirement
    if requirement.startswith(">="):
        req_version = parse_version(requirement[2:])
        cur_version = parse_version(current)
        return cur_version >= req_version

    elif requirement.startswith("~="):
        req_version = parse_version(requirement[2:])
        cur_version = parse_version(current)
        # 兼容版本：主版本和次版本相同
        return (cur_version[0] == req_version[0] and
                cur_version[1] == req_version[1] and
                cur_version[2] >= req_version[2])

    elif requirement.startswith("=="):
        req_version = parse_version(requirement[2:])
        cur_version = parse_version(current)
        return cur_version == req_version

    else:
        # 默认为 >=
        req_version = parse_version(requirement)
        cur_version = parse_version(current)
        return cur_version >= req_version

class VersionChecker:
    """引擎版本检查器"""

    def __init__(self, engine_version: str = "1.0.0"):
        self.engine_version = engine_version

    def check_plugin_compatibility(self, manifest: 'PluginManifest') -> Tuple[bool, str]:
        """
        检查插件兼容性

        Returns:
            (is_compatible, reason)
        """
        try:
            if version_compatible(self.engine_version, manifest.engine):
                return True, "Compatible"
            else:
                return False, f"Engine version {self.engine_version} does not meet requirement {manifest.engine}"
        except ValueError as e:
            return False, f"Invalid version format: {e}"
```

**集成到PluginBus**：
```python
class PluginBus:
    def __init__(self, ..., engine_version: str = "1.0.0"):
        # ... 现有代码
        self.version_checker = VersionChecker(engine_version)

    def compile(self, fail_fast: bool = False):
        # ... 现有检查

        # 新增：版本兼容性检查
        for name, handle in self._handles.items():
            compatible, reason = self.version_checker.check_plugin_compatibility(handle.manifest)
            if not compatible:
                msg = f"Plugin '{name}' version incompatible: {reason}"
                if fail_fast:
                    raise ValueError(msg)
                else:
                    self.logger.warning(msg)
```

---

## 🎯 阶段二：APX-插件自动加载

### 任务2.1: 创建能力映射配置 (P1)

**新文件**：`apt_model/console/capability_plugin_map.py`

```python
"""
能力到插件映射

定义模型能力与推荐插件的对应关系
"""

# 能力 → 插件名称映射
CAPABILITY_PLUGIN_MAP = {
    # MoE模型
    "moe": [
        "route_optimizer",      # 路由优化插件
    ],

    # RAG模型
    "rag": [
        # 待添加RAG专用插件
    ],

    # RL训练的模型
    "rl": [
        "grpo",                 # Group Relative Policy Optimization
    ],

    # 安全/审核模型
    "safety": [
        # 待添加安全审核插件
    ],

    # 量化/蒸馏模型
    "quantization": [
        "model_distillation",   # 蒸馏插件
        "model_pruning",        # 剪枝插件
    ],

    # TVA/VFT模型
    "tva": [
        # VFT模型可能受益于特殊优化
    ],
}

# 反向映射：插件 → 需要的能力
PLUGIN_CAPABILITY_REQUIREMENTS = {
    "route_optimizer": {
        "required": ["moe"],    # 必需MoE能力
        "optional": [],
    },
    "grpo": {
        "required": ["rl"],     # 必需RL能力
        "optional": [],
    },
    "model_distillation": {
        "required": [],
        "optional": ["quantization"],  # 量化模型可选
    },
    "model_pruning": {
        "required": [],
        "optional": ["quantization"],
    },
}

def get_recommended_plugins(capabilities: List[str]) -> List[str]:
    """
    根据模型能力获取推荐插件列表

    Args:
        capabilities: 模型能力列表

    Returns:
        推荐的插件名称列表（去重）
    """
    plugins = []
    for cap in capabilities:
        if cap in CAPABILITY_PLUGIN_MAP:
            plugins.extend(CAPABILITY_PLUGIN_MAP[cap])

    return list(set(plugins))  # 去重

def check_plugin_requirements(plugin_name: str, capabilities: List[str]) -> Tuple[bool, str]:
    """
    检查插件的能力需求是否满足

    Args:
        plugin_name: 插件名称
        capabilities: 模型能力列表

    Returns:
        (is_satisfied, reason)
    """
    if plugin_name not in PLUGIN_CAPABILITY_REQUIREMENTS:
        return True, "No specific requirements"

    reqs = PLUGIN_CAPABILITY_REQUIREMENTS[plugin_name]

    # 检查必需能力
    for required_cap in reqs["required"]:
        if required_cap not in capabilities:
            return False, f"Missing required capability: {required_cap}"

    return True, "Requirements satisfied"
```

---

### 任务2.2: 创建自动插件加载器 (P1)

**新文件**：`apt_model/console/auto_loader.py`

```python
"""
自动插件加载器

基于模型能力自动加载和配置插件
"""

from typing import List, Dict, Optional, Any
from pathlib import Path
import logging

from apt_model.console.plugin_standards import PluginBase, PluginManifest
from apt_model.console.capability_plugin_map import (
    get_recommended_plugins,
    check_plugin_requirements,
)
from apt_model.tools.apx.detectors import detect_capabilities

logger = logging.getLogger(__name__)

class AutoPluginLoader:
    """自动插件加载器"""

    def __init__(self, plugin_registry: Dict[str, type]):
        """
        Args:
            plugin_registry: 插件注册表 {name: PluginClass}
        """
        self.plugin_registry = plugin_registry

    def analyze_model(self, model_path: Path) -> Dict[str, Any]:
        """
        分析模型并生成插件建议

        Args:
            model_path: 模型目录路径

        Returns:
            分析结果字典
        """
        # 检测能力
        capabilities = detect_capabilities(model_path)

        # 获取推荐插件
        recommended = get_recommended_plugins(capabilities)

        # 检查每个插件的需求
        available_plugins = []
        unavailable_plugins = []

        for plugin_name in recommended:
            satisfied, reason = check_plugin_requirements(plugin_name, capabilities)

            if satisfied and plugin_name in self.plugin_registry:
                available_plugins.append({
                    "name": plugin_name,
                    "reason": reason,
                })
            else:
                unavailable_plugins.append({
                    "name": plugin_name,
                    "reason": reason if not satisfied else "Plugin not registered",
                })

        return {
            "capabilities": capabilities,
            "recommended_plugins": recommended,
            "available_plugins": available_plugins,
            "unavailable_plugins": unavailable_plugins,
        }

    def load_for_model(
        self,
        model_path: Path,
        auto_enable: bool = True,
        dry_run: bool = False,
    ) -> List[PluginBase]:
        """
        为模型自动加载插件

        Args:
            model_path: 模型目录路径
            auto_enable: 是否自动启用推荐插件
            dry_run: 仅分析不加载

        Returns:
            已加载的插件实例列表
        """
        analysis = self.analyze_model(model_path)

        logger.info(f"Model capabilities detected: {analysis['capabilities']}")
        logger.info(f"Recommended plugins: {analysis['recommended_plugins']}")

        if dry_run:
            return []

        loaded_plugins = []

        if auto_enable:
            for plugin_info in analysis['available_plugins']:
                plugin_name = plugin_info['name']

                try:
                    # 实例化插件
                    plugin_class = self.plugin_registry[plugin_name]
                    plugin = plugin_class()
                    loaded_plugins.append(plugin)

                    logger.info(f"Auto-loaded plugin: {plugin_name}")

                except Exception as e:
                    logger.error(f"Failed to load plugin {plugin_name}: {e}")

        return loaded_plugins
```

---

### 任务2.3: 创建APX加载器 (P2)

**新文件**：`apt_model/console/apx_loader.py`

```python
"""
APX包加载器

从APX包加载模型并自动配置插件
"""

import zipfile
import tempfile
import shutil
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
import logging
import yaml

from apt_model.tools.apx.detectors import detect_capabilities

logger = logging.getLogger(__name__)

class APXLoader:
    """APX包加载器"""

    def __init__(self, extract_dir: Optional[Path] = None):
        """
        Args:
            extract_dir: APX包解压目录，None则使用临时目录
        """
        self.extract_dir = extract_dir
        self._temp_dirs = []  # 跟踪临时目录以便清理

    def load(self, apx_path: Path) -> Dict[str, Any]:
        """
        加载APX包

        Args:
            apx_path: APX文件路径

        Returns:
            {
                "manifest": apx.yaml内容,
                "artifacts_dir": 解压的artifacts目录,
                "adapters_dir": 解压的adapters目录,
                "capabilities": 检测到的能力,
                "extract_dir": 解压根目录,
            }
        """
        if not apx_path.exists():
            raise FileNotFoundError(f"APX file not found: {apx_path}")

        # 确定解压目录
        if self.extract_dir is None:
            extract_root = Path(tempfile.mkdtemp(prefix="apx_"))
            self._temp_dirs.append(extract_root)
        else:
            extract_root = self.extract_dir / apx_path.stem
            extract_root.mkdir(parents=True, exist_ok=True)

        logger.info(f"Extracting APX to: {extract_root}")

        # 解压APX包
        with zipfile.ZipFile(apx_path, 'r') as zf:
            zf.extractall(extract_root)

        # 读取manifest
        manifest_path = extract_root / "apx.yaml"
        if not manifest_path.exists():
            raise ValueError(f"Invalid APX: apx.yaml not found")

        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = yaml.safe_load(f)

        # 检测能力
        artifacts_dir = extract_root / "artifacts"
        if artifacts_dir.exists():
            capabilities = detect_capabilities(artifacts_dir)
        else:
            capabilities = manifest.get("capabilities", {}).get("provides", [])

        logger.info(f"APX capabilities: {capabilities}")

        return {
            "manifest": manifest,
            "artifacts_dir": artifacts_dir,
            "adapters_dir": extract_root / "model" / "adapters",
            "capabilities": capabilities,
            "extract_dir": extract_root,
        }

    def load_with_auto_plugins(
        self,
        apx_path: Path,
        auto_loader: 'AutoPluginLoader',
    ) -> Tuple[Dict[str, Any], List['PluginBase']]:
        """
        加载APX包并自动配置插件

        Args:
            apx_path: APX文件路径
            auto_loader: 自动插件加载器

        Returns:
            (apx_info, loaded_plugins)
        """
        # 加载APX
        apx_info = self.load(apx_path)

        # 基于capabilities自动加载插件
        # 使用artifacts目录作为模型路径
        plugins = auto_loader.load_for_model(
            apx_info['artifacts_dir'],
            auto_enable=True,
        )

        logger.info(f"Auto-loaded {len(plugins)} plugins for APX model")

        return apx_info, plugins

    def cleanup(self):
        """清理临时目录"""
        for temp_dir in self._temp_dirs:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
                logger.debug(f"Cleaned up temp directory: {temp_dir}")
        self._temp_dirs.clear()

    def __del__(self):
        """析构时自动清理"""
        self.cleanup()
```

---

## 🎯 阶段三：现有插件适配

### 任务3.1: 创建插件适配器基类 (P2)

**新文件**：`apt_model/console/plugin_adapter.py`

```python
"""
插件适配器

将现有独立插件适配到新的PluginBase系统
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import logging

from apt_model.console.plugin_standards import PluginBase, PluginManifest, PluginPriority

logger = logging.getLogger(__name__)

class LegacyPluginAdapter(PluginBase):
    """
    遗留插件适配器

    将现有的独立插件类包装为符合PluginBase的插件
    """

    def __init__(
        self,
        legacy_plugin: Any,
        name: str,
        priority: int,
        events: List[str],
        **manifest_kwargs
    ):
        """
        Args:
            legacy_plugin: 原有插件实例
            name: 插件名称
            priority: 优先级
            events: 监听的事件列表
            **manifest_kwargs: 其他manifest参数
        """
        self.legacy_plugin = legacy_plugin
        self._name = name
        self._priority = priority
        self._events = events
        self._manifest_kwargs = manifest_kwargs

    def get_manifest(self) -> PluginManifest:
        """返回插件manifest"""
        return PluginManifest(
            name=self._name,
            priority=self._priority,
            events=self._events,
            **self._manifest_kwargs
        )

    def _call_legacy_method(self, method_name: str, context: Dict[str, Any]):
        """调用遗留插件的方法"""
        if hasattr(self.legacy_plugin, method_name):
            method = getattr(self.legacy_plugin, method_name)
            try:
                return method(context)
            except Exception as e:
                logger.error(f"Error calling {self._name}.{method_name}: {e}")
                raise
        else:
            logger.warning(f"Legacy plugin {self._name} has no method {method_name}")

    # 实现所有事件方法
    def on_init(self, context: Dict[str, Any]):
        self._call_legacy_method('on_init', context)

    def on_batch_start(self, context: Dict[str, Any]):
        self._call_legacy_method('on_batch_start', context)

    def on_batch_end(self, context: Dict[str, Any]):
        self._call_legacy_method('on_batch_end', context)

    def on_step_start(self, context: Dict[str, Any]):
        self._call_legacy_method('on_step_start', context)

    def on_step_end(self, context: Dict[str, Any]):
        self._call_legacy_method('on_step_end', context)

    def on_decode(self, context: Dict[str, Any]):
        self._call_legacy_method('on_decode', context)

    def on_shutdown(self, context: Dict[str, Any]):
        self._call_legacy_method('on_shutdown', context)
```

---

### 任务3.2: 创建8个插件的适配器 (P2)

为每个现有插件创建适配配置：

**新文件**：`apt_model/console/plugins/legacy_adapters.py`

```python
"""
遗留插件适配器配置

为8个现有插件提供适配配置
"""

from apt_model.console.plugin_adapter import LegacyPluginAdapter
from apt_model.console.plugin_standards import PluginPriority, PluginEvent

# 适配器工厂函数

def create_huggingface_adapter(hf_plugin):
    """HuggingFace集成插件适配器"""
    return LegacyPluginAdapter(
        legacy_plugin=hf_plugin,
        name="huggingface_integration",
        priority=PluginPriority.ADMIN_AUDIT,  # 700
        events=[PluginEvent.ON_INIT, PluginEvent.ON_SHUTDOWN],
        category="integration",
        required_capabilities=[],
        optional_capabilities=[],
    )

def create_cloud_storage_adapter(cloud_plugin):
    """云存储插件适配器"""
    return LegacyPluginAdapter(
        legacy_plugin=cloud_plugin,
        name="cloud_storage",
        priority=PluginPriority.ADMIN_AUDIT,  # 700
        events=[PluginEvent.ON_BATCH_END, PluginEvent.ON_SHUTDOWN],
        category="storage",
        required_capabilities=[],
        optional_capabilities=[],
    )

def create_ollama_export_adapter(ollama_plugin):
    """Ollama导出插件适配器"""
    return LegacyPluginAdapter(
        legacy_plugin=ollama_plugin,
        name="ollama_export",
        priority=PluginPriority.POST_CLEANUP,  # 900
        events=[PluginEvent.ON_SHUTDOWN],
        category="export",
        required_capabilities=[],
        optional_capabilities=["quantization"],
    )

def create_distillation_adapter(distill_plugin):
    """模型蒸馏插件适配器"""
    return LegacyPluginAdapter(
        legacy_plugin=distill_plugin,
        name="model_distillation",
        priority=PluginPriority.TRAINING,  # 350
        events=[PluginEvent.ON_BATCH_END, PluginEvent.ON_STEP_END],
        category="training",
        required_capabilities=[],
        optional_capabilities=["quantization"],
    )

def create_pruning_adapter(prune_plugin):
    """模型剪枝插件适配器"""
    return LegacyPluginAdapter(
        legacy_plugin=prune_plugin,
        name="model_pruning",
        priority=PluginPriority.TRAINING,  # 350
        events=[PluginEvent.ON_BATCH_END, PluginEvent.ON_STEP_END],
        category="training",
        required_capabilities=[],
        optional_capabilities=["quantization"],
    )

def create_multimodal_adapter(mm_plugin):
    """多模态训练插件适配器"""
    return LegacyPluginAdapter(
        legacy_plugin=mm_plugin,
        name="multimodal_training",
        priority=PluginPriority.TRAINING,  # 350
        events=[PluginEvent.ON_BATCH_START, PluginEvent.ON_BATCH_END],
        category="training",
        required_capabilities=[],
        optional_capabilities=[],
    )

def create_data_processors_adapter(data_plugin):
    """数据处理插件适配器"""
    return LegacyPluginAdapter(
        legacy_plugin=data_plugin,
        name="data_processors",
        priority=PluginPriority.CORE_RUNTIME,  # 100
        events=[PluginEvent.ON_INIT],
        category="data",
        required_capabilities=[],
        optional_capabilities=[],
    )

def create_debugging_adapter(debug_plugin):
    """高级调试插件适配器"""
    return LegacyPluginAdapter(
        legacy_plugin=debug_plugin,
        name="advanced_debugging",
        priority=PluginPriority.TELEMETRY,  # 800
        events=[
            PluginEvent.ON_BATCH_START,
            PluginEvent.ON_BATCH_END,
            PluginEvent.ON_STEP_START,
            PluginEvent.ON_STEP_END,
        ],
        category="debug",
        required_capabilities=[],
        optional_capabilities=[],
    )
```

---

## 🎯 阶段四：Console Core集成

### 任务4.1: 更新Console Core (P1)

在`apt_model/console/core.py`中添加自动加载功能：

```python
from apt_model.console.auto_loader import AutoPluginLoader
from apt_model.console.apx_loader import APXLoader
from apt_model.console.version_checker import VersionChecker

class ConsoleCore:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # ... 现有代码

        # 新增：版本检查器
        engine_version = config.get('engine_version', '1.0.0')
        self.version_checker = VersionChecker(engine_version)

        # 新增：自动插件加载器
        self.auto_loader = AutoPluginLoader(self.plugin_registry)

        # 新增：APX加载器
        self.apx_loader = APXLoader()

    def load_apx_model(
        self,
        apx_path: Path,
        auto_configure_plugins: bool = True,
    ) -> Dict[str, Any]:
        """
        加载APX模型并自动配置插件

        Args:
            apx_path: APX文件路径
            auto_configure_plugins: 是否自动配置插件

        Returns:
            APX信息字典
        """
        if auto_configure_plugins:
            apx_info, plugins = self.apx_loader.load_with_auto_plugins(
                apx_path,
                self.auto_loader,
            )

            # 注册自动加载的插件
            for plugin in plugins:
                self.register_plugin(plugin)

            logger.info(f"Auto-registered {len(plugins)} plugins for APX model")
        else:
            apx_info = self.apx_loader.load(apx_path)

        return apx_info

    def analyze_model_for_plugins(
        self,
        model_path: Path,
        dry_run: bool = True,
    ) -> Dict[str, Any]:
        """
        分析模型并推荐插件（不加载）

        Args:
            model_path: 模型目录路径
            dry_run: 是否仅分析

        Returns:
            分析结果
        """
        return self.auto_loader.analyze_model(model_path)
```

---

## 📁 文件清单

### 新增文件

| 文件 | 描述 | 优先级 |
|------|------|--------|
| `apt_model/console/version_checker.py` | 版本兼容性检查 | P2 |
| `apt_model/console/capability_plugin_map.py` | 能力-插件映射 | P1 |
| `apt_model/console/auto_loader.py` | 自动插件加载器 | P1 |
| `apt_model/console/apx_loader.py` | APX包加载器 | P2 |
| `apt_model/console/plugin_adapter.py` | 插件适配器基类 | P2 |
| `apt_model/console/plugins/legacy_adapters.py` | 遗留插件适配器 | P2 |

### 修改文件

| 文件 | 修改内容 | 优先级 |
|------|---------|--------|
| `apt_model/__init__.py` | 延迟导入torch依赖 | P1 |
| `apt_model/console/plugin_standards.py` | 扩展PluginManifest | P2 |
| `apt_model/console/plugin_bus.py` | 集成版本检查 | P2 |
| `apt_model/console/core.py` | 添加自动加载功能 | P1 |

---

## 🧪 测试计划

### 单元测试

1. **版本检查测试**
   - `test_version_parser.py`
   - `test_version_compatibility.py`

2. **能力映射测试**
   - `test_capability_mapping.py`
   - `test_plugin_requirements.py`

3. **自动加载器测试**
   - `test_auto_loader_analyze.py`
   - `test_auto_loader_load.py`

4. **APX加载器测试**
   - `test_apx_loader.py`
   - `test_apx_auto_plugins.py`

### 集成测试

1. **完整工作流测试**
   ```python
   # 测试：加载APX → 检测能力 → 自动加载插件
   apx_path = Path("test_models/mixtral-moe.apx")
   core = ConsoleCore()
   apx_info = core.load_apx_model(apx_path, auto_configure_plugins=True)
   assert 'moe' in apx_info['capabilities']
   assert any(p.get_manifest().name == 'route_optimizer' for p in core.plugin_bus._handles.values())
   ```

2. **插件适配器测试**
   ```python
   # 测试：遗留插件适配
   from legacy_plugins import HuggingFacePlugin
   from apt_model.console.plugins.legacy_adapters import create_huggingface_adapter

   legacy = HuggingFacePlugin()
   adapted = create_huggingface_adapter(legacy)
   assert isinstance(adapted, PluginBase)
   ```

---

## 📊 实施时间表

| 阶段 | 任务 | 预计时间 | 优先级 |
|------|------|---------|--------|
| **阶段一** | 核心系统增强 | 2-3小时 | P1-P2 |
| 1.1 | 修复__init__.py | 30分钟 | P1 |
| 1.2 | 扩展PluginManifest | 1小时 | P2 |
| 1.3 | 版本检测系统 | 1-1.5小时 | P2 |
| **阶段二** | APX-插件自动加载 | 3-4小时 | P1-P2 |
| 2.1 | 能力映射配置 | 30分钟 | P1 |
| 2.2 | 自动插件加载器 | 1.5-2小时 | P1 |
| 2.3 | APX加载器 | 1-1.5小时 | P2 |
| **阶段三** | 现有插件适配 | 2-3小时 | P2 |
| 3.1 | 适配器基类 | 1小时 | P2 |
| 3.2 | 8个插件适配器 | 1-2小时 | P2 |
| **阶段四** | Console Core集成 | 1-2小时 | P1 |
| 4.1 | 更新Console Core | 1-2小时 | P1 |
| **测试** | 单元+集成测试 | 2-3小时 | P1 |

**总计**: ~10-15小时

---

## ✅ 验收标准

### 功能验收

- [ ] APX包可以自动检测能力
- [ ] 基于能力自动推荐插件
- [ ] 自动加载推荐插件
- [ ] 版本兼容性检查正常工作
- [ ] 8个遗留插件成功适配
- [ ] Console Core集成完整

### 性能验收

- [ ] APX加载时间 < 1秒
- [ ] 能力检测时间 < 100ms
- [ ] 插件自动加载时间 < 500ms

### 代码质量验收

- [ ] 所有新代码有类型注解
- [ ] 所有新代码有文档字符串
- [ ] 单元测试覆盖率 > 80%
- [ ] 集成测试覆盖主要工作流

---

## 📝 注意事项

1. **向后兼容性**: 所有改动必须保持向后兼容
2. **错误处理**: 自动加载失败不应影响系统启动
3. **日志记录**: 所有自动化操作都要有详细日志
4. **配置选项**: 提供开关控制自动加载行为
5. **文档更新**: 更新用户文档和API文档

---

**计划创建完成！准备开始实施。**
