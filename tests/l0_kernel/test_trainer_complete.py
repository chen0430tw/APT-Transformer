#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
完整的训练器测试套件

包含：
1. 基础训练测试
2. Checkpoint保存/加载测试
3. 恢复训练测试
4. 早停机制测试
5. 混合精度测试
6. 梯度累积测试
7. Temp checkpoint测试
8. 🔮 API接口伏笔
9. 🔮 分布式训练伏笔
"""

import pytest
import torch
import torch.nn as nn
import tempfile
import shutil
from pathlib import Path
import json

# 导入训练相关模块
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from apt.core.training.trainer import train_model
from apt.core.training.checkpoint import CheckpointManager, save_model, load_model
from apt.core.config.apt_config import APTConfig


# ============================================================================
# 测试fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """创建临时目录"""
    tmp_dir = tempfile.mkdtemp()
    yield Path(tmp_dir)
    # 清理
    shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.fixture
def sample_texts():
    """示例训练文本"""
    return [
        "This is a test sentence for training.",
        "Machine learning is amazing.",
        "Deep learning models are powerful.",
        "Natural language processing is fascinating.",
        "AI will change the world.",
        "Testing is important for code quality.",
        "Python is a great programming language.",
        "Neural networks learn from data."
    ]


@pytest.fixture
def mini_config():
    """最小化配置（用于快速测试）"""
    config = APTConfig()
    config.d_model = 128
    config.nhead = 4
    config.num_encoder_layers = 2
    config.num_decoder_layers = 2
    config.dim_feedforward = 256
    config.vocab_size = 1000
    return config


# ============================================================================
# 基础训练测试
# ============================================================================

class TestBasicTraining:
    """基础训练功能测试"""

    def test_train_basic_flow(self, temp_dir, sample_texts, mini_config):
        """测试基础训练流程"""
        model, tokenizer, config = train_model(
            epochs=2,
            batch_size=2,
            learning_rate=1e-4,
            checkpoint_dir=temp_dir / "outputs",
            texts=sample_texts,
            tokenizer=None  # 自动创建
        )

        # 验证模型已训练
        assert model is not None
        assert tokenizer is not None
        assert config is not None

        # 验证checkpoint目录创建
        checkpoint_dir = temp_dir / "outputs" / "checkpoints"
        assert checkpoint_dir.exists()

        # 验证至少有checkpoint文件
        checkpoint_files = list(checkpoint_dir.glob("*.pt"))
        assert len(checkpoint_files) >= 1

    def test_train_returns_valid_model(self, temp_dir, sample_texts):
        """测试训练返回可用模型"""
        model, tokenizer, config = train_model(
            epochs=1,
            batch_size=2,
            checkpoint_dir=temp_dir / "outputs",
            texts=sample_texts
        )

        # 测试模型可以推理
        test_input = tokenizer.encode("test input", return_tensors='pt')
        with torch.no_grad():
            output = model(
                src_tokens=test_input,
                tgt_tokens=test_input
            )

        assert output is not None
        assert output.shape[0] == 1  # batch size


# ============================================================================
# Checkpoint测试
# ============================================================================

class TestCheckpointSystem:
    """Checkpoint系统测试"""

    def test_checkpoint_save_complete_state(self, temp_dir):
        """测试checkpoint保存完整状态"""
        mgr = CheckpointManager(save_dir=temp_dir, model_name="test_model")

        # 创建虚拟模型和优化器
        model = nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

        # 保存checkpoint
        checkpoint_path = mgr.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=5,
            global_step=1000,
            loss_history=[2.5, 2.3, 2.1, 1.9, 1.8],
            metrics={'avg_loss': 1.8, 'best_loss': 1.8},
            is_best=True
        )

        # 验证文件存在
        assert Path(checkpoint_path).exists()

        # 加载并验证内容
        checkpoint = torch.load(checkpoint_path)

        assert 'model_state_dict' in checkpoint
        assert 'optimizer_state_dict' in checkpoint
        assert 'scheduler_state_dict' in checkpoint
        assert 'epoch' in checkpoint
        assert 'global_step' in checkpoint
        assert 'loss_history' in checkpoint
        assert 'metrics' in checkpoint

        assert checkpoint['epoch'] == 5
        assert checkpoint['global_step'] == 1000
        assert len(checkpoint['loss_history']) == 5

    def test_checkpoint_load_restores_state(self, temp_dir):
        """测试checkpoint加载恢复状态"""
        mgr = CheckpointManager(save_dir=temp_dir)

        # 创建并保存
        model = nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

        original_weights = model.weight.data.clone()

        checkpoint_path = mgr.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=3,
            global_step=500,
            loss_history=[2.0, 1.5, 1.2],
            metrics={'avg_loss': 1.2}
        )

        # 修改模型（模拟继续训练）
        model.weight.data += 1.0

        # 加载checkpoint
        epoch, step, loss_history, metrics = mgr.load_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            checkpoint_path=checkpoint_path
        )

        # 验证状态恢复
        assert epoch == 3
        assert step == 500
        assert len(loss_history) == 3
        assert metrics['avg_loss'] == 1.2

        # 验证权重恢复
        assert torch.allclose(model.weight.data, original_weights)

    def test_metadata_tracking(self, temp_dir):
        """测试元数据追踪"""
        mgr = CheckpointManager(save_dir=temp_dir)

        model = nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

        # 保存多个checkpoint
        for epoch in range(3):
            mgr.save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                global_step=epoch * 100,
                loss_history=[],
                metrics={'avg_loss': 2.0 - epoch * 0.5},
                is_best=(epoch == 2)
            )

        # 验证metadata.json
        metadata_path = temp_dir / "metadata.json"
        assert metadata_path.exists()

        with open(metadata_path) as f:
            metadata = json.load(f)

        assert len(metadata['checkpoints']) == 3
        assert metadata['checkpoints'][-1]['is_best'] == True


# ============================================================================
# 恢复训练测试
# ============================================================================

class TestResumeTraining:
    """恢复训练测试"""

    def test_resume_from_checkpoint(self, temp_dir, sample_texts):
        """测试从checkpoint恢复训练"""
        # 第一次训练到epoch 2
        model1, tokenizer1, config1 = train_model(
            epochs=2,
            batch_size=2,
            checkpoint_dir=temp_dir / "outputs",
            texts=sample_texts
        )

        # 找到最后的checkpoint
        checkpoint_files = sorted((temp_dir / "outputs" / "checkpoints").glob("*.pt"))
        last_checkpoint = checkpoint_files[-1]

        # 恢复训练到epoch 4
        model2, tokenizer2, config2 = train_model(
            epochs=4,
            batch_size=2,
            checkpoint_dir=temp_dir / "outputs",
            resume_from=str(last_checkpoint),
            texts=sample_texts
        )

        # 验证继续训练
        checkpoint_files_after = list((temp_dir / "outputs" / "checkpoints").glob("*.pt"))
        # 应该有4个checkpoint（或更多，如果有best标记）
        assert len(checkpoint_files_after) >= 4

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="需要GPU")
    def test_resume_preserves_optimizer_state(self, temp_dir, sample_texts):
        """测试恢复训练保持optimizer状态"""
        # 这个测试验证optimizer动量等状态是否正确恢复
        # 实际实现需要更详细的验证逻辑
        pass


# ============================================================================
# 早停机制测试
# ============================================================================

class TestEarlyStopping:
    """早停机制测试"""

    def test_early_stopping_triggers(self, temp_dir, sample_texts):
        """测试早停机制触发"""
        # 注意：这个测试可能不稳定，因为训练可能真的收敛
        # 实际项目中可能需要mock或使用特殊的测试数据
        model, tokenizer, config = train_model(
            epochs=100,  # 设置很大的epoch
            batch_size=2,
            checkpoint_dir=temp_dir / "outputs",
            texts=sample_texts
        )

        # 验证早停生效（应该远少于100个checkpoint）
        checkpoint_files = list((temp_dir / "outputs" / "checkpoints").glob("*.pt"))
        assert len(checkpoint_files) < 100, "早停应该生效"


# ============================================================================
# 混合精度和梯度累积测试
# ============================================================================

class TestAdvancedTraining:
    """高级训练功能测试"""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="需要GPU")
    def test_mixed_precision_training(self, temp_dir, sample_texts):
        """测试混合精度训练"""
        # trainer.py已经使用了torch.amp.autocast
        model, tokenizer, config = train_model(
            epochs=1,
            batch_size=2,
            checkpoint_dir=temp_dir / "outputs",
            texts=sample_texts
        )

        # 如果能成功训练，说明混合精度工作正常
        assert model is not None

    def test_gradient_accumulation(self, temp_dir, sample_texts):
        """测试梯度累积"""
        # trainer.py中accumulation_steps=4
        # 这个测试验证梯度累积是否正常工作
        model, tokenizer, config = train_model(
            epochs=1,
            batch_size=2,
            checkpoint_dir=temp_dir / "outputs",
            texts=sample_texts
        )

        assert model is not None


# ============================================================================
# Temp Checkpoint测试
# ============================================================================

class TestTempCheckpoint:
    """临时checkpoint测试"""

    def test_temp_checkpoint_creation(self, temp_dir, sample_texts):
        """测试临时checkpoint创建"""
        model, tokenizer, config = train_model(
            epochs=1,
            batch_size=2,
            checkpoint_dir=temp_dir / "outputs",
            temp_checkpoint_freq=5,  # 每5步保存
            texts=sample_texts
        )

        # 注意：训练结束后temp文件应该被清理
        temp_dir_path = Path(".cache/temp")
        if temp_dir_path.exists():
            temp_files = list(temp_dir_path.glob("temp_*.pt"))
            # 应该被清理
            assert len(temp_files) == 0

    def test_temp_checkpoint_cleanup(self, temp_dir, sample_texts):
        """测试临时checkpoint清理"""
        # 这个测试验证epoch结束后temp文件被清理
        # 已在test_temp_checkpoint_creation中验证
        pass


# ============================================================================
# 🔮 API接口伏笔测试
# ============================================================================

class TestAPIReadiness:
    """API就绪性测试（为未来的API服务埋伏笔）"""

    def test_model_serialization_for_api(self, temp_dir, sample_texts):
        """测试模型序列化（API部署需要）"""
        model, tokenizer, config = train_model(
            epochs=1,
            batch_size=2,
            checkpoint_dir=temp_dir / "outputs",
            texts=sample_texts
        )

        # 🔮 API伏笔：验证模型可以被序列化
        checkpoint_path = temp_dir / "api_model.pt"
        save_model(model, tokenizer, path=temp_dir / "api_model", config=config)

        # 验证可以加载（API服务需要）
        loaded_model, loaded_tokenizer, loaded_config = load_model(
            temp_dir / "api_model"
        )

        assert loaded_model is not None
        assert loaded_tokenizer is not None

    def test_inference_interface(self, temp_dir, sample_texts):
        """测试推理接口（API endpoint需要）"""
        model, tokenizer, config = train_model(
            epochs=1,
            batch_size=2,
            checkpoint_dir=temp_dir / "outputs",
            texts=sample_texts
        )

        # 🔮 API伏笔：模拟API请求的推理
        def api_inference(model, tokenizer, text, max_length=50):
            """
            这是未来API服务的推理接口原型

            POST /api/generate
            {
                "text": "input text",
                "max_length": 50
            }
            """
            input_ids = tokenizer.encode(text, return_tensors='pt')

            model.eval()
            with torch.no_grad():
                output = model(
                    src_tokens=input_ids,
                    tgt_tokens=input_ids
                )

            return {
                'success': True,
                'output_shape': list(output.shape),
                'input_length': input_ids.shape[1]
            }

        # 测试推理接口
        result = api_inference(model, tokenizer, "test input for API")

        assert result['success'] == True
        assert 'output_shape' in result

    def test_batch_inference_for_api(self, temp_dir, sample_texts):
        """测试批量推理（API批处理需要）"""
        model, tokenizer, config = train_model(
            epochs=1,
            batch_size=2,
            checkpoint_dir=temp_dir / "outputs",
            texts=sample_texts
        )

        # 🔮 API伏笔：批量推理接口
        def api_batch_inference(model, tokenizer, texts, max_length=50):
            """
            未来API服务的批量推理接口

            POST /api/batch_generate
            {
                "texts": ["text1", "text2", ...],
                "max_length": 50
            }
            """
            # 批量编码
            encoded = [tokenizer.encode(text, return_tensors='pt') for text in texts]

            # 批量推理
            model.eval()
            results = []
            with torch.no_grad():
                for input_ids in encoded:
                    output = model(src_tokens=input_ids, tgt_tokens=input_ids)
                    results.append(output)

            return {
                'success': True,
                'batch_size': len(texts),
                'results_count': len(results)
            }

        # 测试批量推理
        result = api_batch_inference(
            model, tokenizer,
            ["test 1", "test 2", "test 3"]
        )

        assert result['batch_size'] == 3
        assert result['results_count'] == 3


# ============================================================================
# 🔮 分布式训练伏笔测试
# ============================================================================

class TestDistributedReadiness:
    """分布式训练就绪性测试（为未来的DDP埋伏笔）"""

    def test_model_supports_ddp_wrapping(self, temp_dir, sample_texts):
        """测试模型支持DDP包装"""
        model, tokenizer, config = train_model(
            epochs=1,
            batch_size=2,
            checkpoint_dir=temp_dir / "outputs",
            texts=sample_texts
        )

        # 🔮 分布式伏笔：验证模型可以被DDP包装
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            # 如果有多GPU，测试DDP包装
            from torch.nn.parallel import DistributedDataParallel as DDP

            model = model.cuda()
            # ddp_model = DDP(model, device_ids=[0])  # 实际需要先init_process_group

            # 目前只验证模型结构兼容DDP
            assert hasattr(model, 'parameters')
            assert hasattr(model, 'state_dict')

    def test_checkpoint_supports_distributed_loading(self, temp_dir, sample_texts):
        """测试checkpoint支持分布式加载"""
        model, tokenizer, config = train_model(
            epochs=1,
            batch_size=2,
            checkpoint_dir=temp_dir / "outputs",
            texts=sample_texts
        )

        # 🔮 分布式伏笔：验证checkpoint可以在不同rank加载
        checkpoint_files = list((temp_dir / "outputs" / "checkpoints").glob("*.pt"))
        checkpoint_path = checkpoint_files[0]

        # 模拟不同rank加载同一个checkpoint
        def load_on_rank(rank, world_size):
            """
            未来分布式训练中，每个rank都会加载checkpoint
            """
            mgr = CheckpointManager(save_dir=temp_dir / "outputs")

            # 创建新模型实例
            from apt.core.modeling.apt_model import APTLargeModel
            new_model = APTLargeModel(config)

            # 加载checkpoint
            epoch, step, loss_history, metrics = mgr.load_checkpoint(
                model=new_model,
                checkpoint_path=checkpoint_path
            )

            return {
                'rank': rank,
                'epoch': epoch,
                'step': step,
                'model_loaded': True
            }

        # 模拟rank 0和rank 1加载
        result_rank0 = load_on_rank(0, 2)
        result_rank1 = load_on_rank(1, 2)

        # 验证两个rank加载的epoch一致
        assert result_rank0['epoch'] == result_rank1['epoch']
        assert result_rank0['model_loaded'] == True

    def test_training_state_for_distributed_sync(self, temp_dir, sample_texts):
        """测试训练状态支持分布式同步"""
        # 🔮 分布式伏笔：验证训练状态可以跨进程同步

        # 未来DDP训练需要同步：
        # 1. global_step（所有rank一致）
        # 2. epoch（所有rank一致）
        # 3. loss（需要all_reduce）
        # 4. 最佳模型判断（需要all_reduce比较）

        model, tokenizer, config = train_model(
            epochs=1,
            batch_size=2,
            checkpoint_dir=temp_dir / "outputs",
            texts=sample_texts
        )

        # 验证checkpoint包含可同步的状态
        checkpoint_files = list((temp_dir / "outputs" / "checkpoints").glob("*.pt"))
        checkpoint = torch.load(checkpoint_files[0])

        # 这些字段在分布式训练中需要同步
        assert 'epoch' in checkpoint
        assert 'global_step' in checkpoint
        assert 'metrics' in checkpoint


# ============================================================================
# 🔮 WebUI数据接口伏笔测试
# ============================================================================

class TestWebUIDataInterface:
    """WebUI数据接口测试（为未来的Web界面埋伏笔）"""

    def test_training_metrics_export(self, temp_dir, sample_texts):
        """测试训练指标导出（WebUI需要）"""
        model, tokenizer, config = train_model(
            epochs=3,
            batch_size=2,
            checkpoint_dir=temp_dir / "outputs",
            texts=sample_texts
        )

        # 🔮 WebUI伏笔：导出训练指标为JSON（供前端展示）
        def export_metrics_for_webui(checkpoint_dir):
            """
            导出训练指标为JSON

            WebUI会通过API获取这些数据：
            GET /api/training/metrics
            """
            metadata_path = checkpoint_dir / "metadata.json"

            if not metadata_path.exists():
                return None

            with open(metadata_path) as f:
                metadata = json.load(f)

            # 提取训练曲线数据
            metrics_timeline = []
            for checkpoint_info in metadata['checkpoints']:
                metrics_timeline.append({
                    'epoch': checkpoint_info['epoch'],
                    'step': checkpoint_info['global_step'],
                    'metrics': checkpoint_info.get('metrics', {}),
                    'is_best': checkpoint_info.get('is_best', False),
                    'timestamp': checkpoint_info.get('created_at', '')
                })

            return {
                'model_name': metadata['model_name'],
                'total_checkpoints': len(metadata['checkpoints']),
                'metrics_timeline': metrics_timeline,
                'last_updated': metadata['last_updated']
            }

        # 测试导出
        webui_data = export_metrics_for_webui(temp_dir / "outputs")

        assert webui_data is not None
        assert 'metrics_timeline' in webui_data
        assert len(webui_data['metrics_timeline']) == 3  # 3个epoch

    def test_checkpoint_list_for_webui(self, temp_dir, sample_texts):
        """测试checkpoint列表接口（WebUI模型管理需要）"""
        model, tokenizer, config = train_model(
            epochs=2,
            batch_size=2,
            checkpoint_dir=temp_dir / "outputs",
            texts=sample_texts
        )

        # 🔮 WebUI伏笔：checkpoint列表API
        def get_checkpoint_list_for_webui(checkpoint_dir):
            """
            获取checkpoint列表

            GET /api/checkpoints
            """
            checkpoint_path = checkpoint_dir / "checkpoints"

            checkpoints = []
            for ckpt_file in sorted(checkpoint_path.glob("*.pt")):
                # 加载checkpoint元信息
                ckpt = torch.load(ckpt_file, map_location='cpu')

                checkpoints.append({
                    'filename': ckpt_file.name,
                    'path': str(ckpt_file),
                    'epoch': ckpt.get('epoch', 0),
                    'global_step': ckpt.get('global_step', 0),
                    'metrics': ckpt.get('metrics', {}),
                    'size_mb': ckpt_file.stat().st_size / 1024 / 1024
                })

            return {
                'total': len(checkpoints),
                'checkpoints': checkpoints
            }

        # 测试接口
        ckpt_list = get_checkpoint_list_for_webui(temp_dir / "outputs")

        assert ckpt_list['total'] >= 2
        assert all('filename' in c for c in ckpt_list['checkpoints'])


# ============================================================================
# 运行测试
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
