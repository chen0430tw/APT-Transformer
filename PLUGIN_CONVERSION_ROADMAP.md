# APT-Transformer 插件转换路线图

**Version**: 1.0
**Last Updated**: 2026-01-21
**Status**: Tier 1 Complete ✅

---

## 🎯 总体目标

将APT-Transformer从单体架构转向插件化架构，提升：
- **模块性** - 清晰的功能边界
- **可选性** - 按需加载功能
- **可维护性** - 独立开发和测试
- **部署灵活性** - 多种配置方案

---

## 📊 转换进度

### 总览

| Tier | 描述 | 模块数 | 状态 | 完成度 |
|------|------|-------|------|--------|
| Tier 1 | 高价值，低成本 | 6 | ✅ Complete | 100% |
| Tier 2 | 高价值，中成本 | 8 | 📋 Planned | 0% |
| Tier 3 | 复杂研究特性 | TBD | 🔮 Future | 0% |
| **Total** | - | **14+** | - | **43%** |

**注**: 从原33个候选模块中，移除了不该做插件的（工具、核心模块等）

### 插件生态增长

```
Phase 0 (Legacy):     11 plugins (混乱状态)
Phase 1 (深度重构):   11 plugins → 4 categories (core/integration/distillation/experimental)
Phase 2 (Tier 1):    +6 plugins → +4 categories (monitoring/visualization/evaluation/infrastructure)
Phase 3 (Tier 2):    +10 plugins → +6 categories (export/optimization/rl/data/protocol/retrieval)
Phase 4 (Tier 3):    +17 plugins → 复杂模块
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Final Target:         44 plugins across 14+ categories
```

---

## ✅ Tier 1: 已完成 (6/6)

### 转换列表

#### Monitoring (2/2) ✅
- [x] `gradient_monitor_plugin.py` - 梯度监控
- [x] `resource_monitor_plugin.py` - 资源监控

#### Visualization (1/1) ✅
- [x] `model_visualization_plugin.py` - 模型可视化

#### Evaluation (2/2) ✅
- [x] `model_evaluator_plugin.py` - 模型评估
- [x] `model_comparison_plugin.py` - 模型对比

#### Infrastructure (1/1) ✅
- [x] `logging_plugin.py` - 集中式日志

### 成果
- **提交**: `eee2e11`
- **文件**: 12 files, +4,166 lines
- **新类别**: 4 categories
- **测试**: Pending

---

## 📋 Tier 2: 计划中 (0/8)

**重要**: 不是所有模块都该做插件！
- ❌ APX Converter - 这是打包**工具**，不是插件
- ❌ Data Processor/Pipeline - 核心功能，应保持为**模块**

### Optimization Plugins (0/1)

| Module | Source | Target | Priority | Estimated Effort |
|--------|--------|--------|----------|-----------------|
| MXFP4 Quantization | `apt/perf/optimization/mxfp4_quantization.py` | `optimization/mxfp4_quantization_plugin.py` | High | 6h |

**依赖**: torch, numpy
**测试需求**: 量化精度测试、性能基准

---

### RL Plugins (0/4)

| Module | Source | Target | Priority | Estimated Effort |
|--------|--------|--------|----------|-----------------|
| RLHF Trainer | `apt/apps/rl/rlhf_trainer.py` | `rl/rlhf_trainer_plugin.py` | High | 8h |
| DPO Trainer | `apt/apps/rl/dpo_trainer.py` | `rl/dpo_trainer_plugin.py` | High | 6h |
| GRPO Trainer | `apt/apps/rl/grpo_trainer.py` | `rl/grpo_trainer_plugin.py` | Medium | 6h |
| Reward Model | `apt/apps/rl/reward_model.py` | `rl/reward_model_plugin.py` | Medium | 4h |

**依赖**: torch, transformers, trl
**测试需求**: 对齐质量测试、训练稳定性

---

### Protocol Plugins (0/1)

| Module | Source | Target | Priority | Estimated Effort |
|--------|--------|--------|----------|-----------------|
| MCP Integration | `apt_model/modeling/mcp_integration.py` | `protocol/mcp_integration_plugin.py` | Medium | 8h |

**依赖**: asyncio, aiohttp
**测试需求**: 协议兼容性、异步行为

---

### Retrieval Plugins (0/2)

| Module | Source | Target | Priority | Estimated Effort |
|--------|--------|--------|----------|-----------------|
| RAG Integration | `apt_model/modeling/rag_integration.py` | `retrieval/rag_integration_plugin.py` | Medium | 8h |
| KG+RAG Integration | `apt_model/modeling/kg_rag_integration.py` | `retrieval/kg_rag_integration_plugin.py` | Medium | 10h |

**依赖**: faiss, torch, networkx
**测试需求**: 检索质量、融合效果、性能基准

---

### Tier 2 总计
- **总模块数**: 8 (修正: 从10减少到8)
- **总工时估算**: 56小时
- **优先级分布**: 4 High, 4 Medium
- **移除项**: APX Converter (工具), Data Processor/Pipeline (核心模块)

---

## 🔮 Tier 3: 未来规划 (0/17)

### 复杂模块清单

#### Hardware Emulation (3 modules)
- [ ] Virtual Blackwell Stack - GPU特性仿真
- [ ] NPU Backend - NPU加速支持
- [ ] Cloud NPU Adapter - 云NPU适配

**挑战**: 硬件抽象复杂度、测试环境需求

#### Advanced Optimization (4 modules)
- [ ] GPU Flash Optimization - Triton内核优化
- [ ] Extreme Scale Training - 大规模分布式训练
- [ ] MicroVM Compression - 微虚拟机压缩
- [ ] vGPU Stack - 虚拟GPU管理

**挑战**: 性能敏感、CUDA/Triton专业知识

#### Knowledge Systems (4 modules)
- [ ] Knowledge Graph + RAG - KG+RAG融合
- [ ] AIM Memory System - 分层记忆系统
- [ ] GraphRAG System - 图检索系统
- [ ] Knowledge Graph - 知识图谱管理

**挑战**: 状态管理复杂、图算法优化

#### Multimodal & Data (4 modules)
- [ ] Multimodal Adapter - 多模态适配器
- [ ] Multimodal Dataset - 多模态数据集
- [ ] External Data Loader - 外部数据加载
- [ ] HuggingFace Loader - HF集成

**挑战**: 数据格式多样性、依赖管理

#### UI & Interaction (2 modules)
- [ ] Chat Module - 交互式聊天
- [ ] WebUI Extensions - WebUI扩展

**挑战**: 用户体验一致性、实时性

---

## 🛠️ 实施策略

### Tier 2 执行计划

#### Week 1: Export & Optimization
```bash
# Day 1-2: APX Converter
python scripts/convert_tier2_modules.py --module=apx_converter
pytest tests/plugins/export/

# Day 3-5: MXFP4 Quantization
python scripts/convert_tier2_modules.py --module=mxfp4_quantization
pytest tests/plugins/optimization/
```

#### Week 2: RL Plugins
```bash
# Day 1-2: RLHF Trainer
python scripts/convert_tier2_modules.py --module=rlhf_trainer

# Day 3-4: DPO Trainer
python scripts/convert_tier2_modules.py --module=dpo_trainer

# Day 5: GRPO + Reward Model
python scripts/convert_tier2_modules.py --module=grpo_trainer,reward_model
```

#### Week 3: Data & Protocol
```bash
# Day 1-3: Data Plugins
python scripts/convert_tier2_modules.py --category=data

# Day 4-5: Protocol & Retrieval
python scripts/convert_tier2_modules.py --category=protocol,retrieval
```

### 质量保证

每个转换必须包括:
1. **单元测试** - 覆盖率 ≥ 80%
2. **集成测试** - 与核心系统集成
3. **性能基准** - 对比原始实现
4. **文档更新** - 使用指南和API文档

---

## 📈 成功指标

### Tier 2 目标
- [ ] 10个模块成功转换为插件
- [ ] 6个新插件类别创建
- [ ] 所有测试通过 (覆盖率 ≥ 80%)
- [ ] 性能无回退 (±5%)
- [ ] 文档完整更新

### Tier 3 目标
- [ ] 17个复杂模块转换
- [ ] 性能优化完成 (提升10-20%)
- [ ] 完整的插件市场文档
- [ ] 第三方插件支持

---

## 🎓 经验总结

### Tier 1 教训

#### 做得好
1. **分层规划** - 按复杂度分tier
2. **自动化脚本** - 减少手工错误
3. **文档先行** - PLUGIN_CATALOG.md
4. **向后兼容** - 原始文件保留

#### 需改进
1. **测试覆盖** - 需要增加插件测试
2. **导入更新** - 需要自动化导入重写
3. **性能测试** - 需要基准对比

### Best Practices

```python
# 插件转换检查清单
checklist = {
    "分析": [
        "✓ 确认模块职责清晰",
        "✓ 检查依赖关系",
        "✓ 评估耦合度",
    ],
    "转换": [
        "✓ 创建插件类别",
        "✓ 复制并重构代码",
        "✓ 添加插件元数据",
    ],
    "测试": [
        "✓ 单元测试 (≥80%)",
        "✓ 集成测试",
        "✓ 性能基准",
    ],
    "文档": [
        "✓ API文档",
        "✓ 使用示例",
        "✓ 更新PLUGIN_CATALOG.md",
    ],
}
```

---

## 🔗 相关资源

### 文档
- `PLUGIN_MIGRATION_SUMMARY.md` - 迁移总结
- `apt/apps/plugins/PLUGIN_CATALOG.md` - 插件目录
- `docs/product/PLUGIN_SYSTEM_GUIDE.md` - 系统指南

### 脚本
- `scripts/convert_modules_to_plugins.py` - Tier 1转换
- `scripts/deep_restructure_plugins.py` - 深度重构
- `scripts/check_reverse_dependencies.py` - 依赖检查

### 测试
- `tests/l3_product/test_plugin_system.py` - 插件系统测试
- `tests/integration/` - 集成测试

---

## 📞 获取帮助

### 问题排查
1. 查看 `PLUGIN_MIGRATION_SUMMARY.md`
2. 运行 `pytest tests/l3_product/ -v`
3. 检查 `apt/apps/plugins/PLUGIN_CATALOG.md`

### 开发支持
- GitHub Issues: 报告bug或请求功能
- 文档: `docs/product/PLUGIN_SYSTEM_GUIDE.md`
- 示例: `apt/apps/plugins/experimental/`

---

**Next Steps**:
1. ✅ Complete Tier 1 validation
2. ✅ 修正Tier 2计划（移除不该做插件的模块）
3. 📋 Review plugin vs module principles
4. 🚀 Execute Tier 2 conversion (8 modules)
5. 🔮 Design Tier 3 architecture

**Progress**: 6/14+ modules converted (43%) | 4/10+ categories created

**Key Learning**: 不是所有模块都该做插件！工具保持为工具，核心模块保持为模块。
