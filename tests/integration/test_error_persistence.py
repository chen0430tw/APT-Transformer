#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
错误持久化系统测试

测试ErrorPersistence类的所有功能，包括:
- 错误日志记录
- 错误模式识别
- 统计分析
- 数据导出
"""

import os
import json
import pytest
import tempfile
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta

from apt.apt_model.utils.error_persistence import (
    ErrorPersistence,
    log_training_error,
    get_global_error_persistence
)


@pytest.fixture
def temp_db():
    """临时数据库fixture"""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test_errors.db"
        yield db_path


@pytest.fixture
def error_persistence(temp_db):
    """ErrorPersistence实例fixture"""
    ep = ErrorPersistence(db_path=str(temp_db))
    yield ep
    ep.close()


class TestBasicErrorLogging:
    """测试基础错误日志功能"""

    def test_database_initialization(self, temp_db):
        """测试数据库初始化"""
        ep = ErrorPersistence(db_path=str(temp_db))

        # 检查数据库文件是否创建
        assert temp_db.exists()

        # 检查表是否创建
        conn = sqlite3.connect(str(temp_db))
        cursor = conn.cursor()

        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name IN ('error_logs', 'error_patterns')
        """)
        tables = [row[0] for row in cursor.fetchall()]

        assert 'error_logs' in tables
        assert 'error_patterns' in tables

        conn.close()
        ep.close()

    def test_log_simple_error(self, error_persistence):
        """测试记录简单错误"""
        try:
            raise ValueError("Test error message")
        except ValueError as e:
            error_id = error_persistence.log_error(
                error=e,
                severity=ErrorPersistence.SEVERITY_ERROR,
                module="test_module",
                function="test_function"
            )

        # 验证返回的error_id
        assert error_id > 0

        # 验证错误已记录到数据库
        error_persistence.cursor.execute("""
            SELECT error_type, error_message, module, function
            FROM error_logs WHERE id = ?
        """, (error_id,))

        row = error_persistence.cursor.fetchone()
        assert row[0] == "ValueError"
        assert row[1] == "Test error message"
        assert row[2] == "test_module"
        assert row[3] == "test_function"

    def test_log_error_with_context(self, error_persistence):
        """测试记录带上下文的错误"""
        context = {
            'batch_size': 8,
            'learning_rate': 0.001,
            'gpu_memory': '4GB'
        }

        try:
            raise RuntimeError("Out of memory")
        except RuntimeError as e:
            error_id = error_persistence.log_error(
                error=e,
                severity=ErrorPersistence.SEVERITY_CRITICAL,
                context=context,
                epoch=5,
                global_step=1200
            )

        # 验证上下文信息
        error_persistence.cursor.execute("""
            SELECT context, epoch, global_step FROM error_logs WHERE id = ?
        """, (error_id,))

        row = error_persistence.cursor.fetchone()
        stored_context = json.loads(row[0])

        assert stored_context == context
        assert row[1] == 5
        assert row[2] == 1200

    def test_severity_levels(self, error_persistence):
        """测试不同严重性级别"""
        severities = [
            (ErrorPersistence.SEVERITY_DEBUG, "Debug error"),
            (ErrorPersistence.SEVERITY_INFO, "Info error"),
            (ErrorPersistence.SEVERITY_WARNING, "Warning error"),
            (ErrorPersistence.SEVERITY_ERROR, "Error error"),
            (ErrorPersistence.SEVERITY_CRITICAL, "Critical error"),
        ]

        for severity, message in severities:
            try:
                raise Exception(message)
            except Exception as e:
                error_persistence.log_error(e, severity=severity)

        # 验证所有严重性级别都已记录
        error_persistence.cursor.execute("""
            SELECT DISTINCT severity FROM error_logs ORDER BY severity
        """)

        stored_severities = [row[0] for row in error_persistence.cursor.fetchall()]
        assert stored_severities == [0, 1, 2, 3, 4]


class TestErrorPatternRecognition:
    """测试错误模式识别"""

    def test_identical_errors_create_pattern(self, error_persistence):
        """测试相同错误创建模式"""
        # 记录3次相同的错误
        for i in range(3):
            try:
                x = 1 / 0
            except ZeroDivisionError as e:
                error_persistence.log_error(e)

        # 检查错误模式表
        error_persistence.cursor.execute("""
            SELECT error_type, occurrence_count FROM error_patterns
        """)

        patterns = error_persistence.cursor.fetchall()
        assert len(patterns) == 1
        assert patterns[0][0] == "ZeroDivisionError"
        assert patterns[0][1] == 3

    def test_similar_errors_share_pattern(self, error_persistence):
        """测试相似错误共享模式"""
        # 记录数字不同但模式相同的错误
        for i in range(5):
            try:
                raise ValueError(f"Invalid value: {i}")
            except ValueError as e:
                error_persistence.log_error(e)

        # 应该识别为同一个模式
        error_persistence.cursor.execute("""
            SELECT occurrence_count FROM error_patterns WHERE error_type = 'ValueError'
        """)

        count = error_persistence.cursor.fetchone()[0]
        assert count == 5

    def test_get_error_patterns(self, error_persistence):
        """测试获取错误模式列表"""
        # 创建多个不同频率的错误
        for i in range(10):
            try:
                raise ValueError("Frequent error")
            except ValueError as e:
                error_persistence.log_error(e)

        for i in range(3):
            try:
                raise TypeError("Less frequent error")
            except TypeError as e:
                error_persistence.log_error(e)

        # 获取错误模式（至少出现2次）
        patterns = error_persistence.get_error_patterns(min_occurrences=2)

        assert len(patterns) == 2
        # ValueError应该排在前面（出现次数更多）
        assert patterns[0]['error_type'] == 'ValueError'
        assert patterns[0]['occurrence_count'] == 10
        assert patterns[1]['error_type'] == 'TypeError'
        assert patterns[1]['occurrence_count'] == 3

    def test_mark_pattern_resolved(self, error_persistence):
        """测试标记错误模式为已解决"""
        # 创建错误
        try:
            raise ValueError("Test error")
        except ValueError as e:
            error_persistence.log_error(e)

        # 获取错误哈希
        patterns = error_persistence.get_error_patterns(min_occurrences=1, unresolved_only=False)
        error_hash = patterns[0]['error_hash']

        # 标记为已解决
        error_persistence.mark_pattern_resolved(error_hash)

        # 验证已解决
        patterns_unresolved = error_persistence.get_error_patterns(min_occurrences=1, unresolved_only=True)
        assert len(patterns_unresolved) == 0

        patterns_all = error_persistence.get_error_patterns(min_occurrences=1, unresolved_only=False)
        assert patterns_all[0]['resolved'] == True


class TestErrorStatistics:
    """测试错误统计分析"""

    def test_error_statistics_basic(self, error_persistence):
        """测试基本统计信息"""
        # 创建不同严重性的错误
        errors = [
            (ErrorPersistence.SEVERITY_WARNING, ValueError("Warning 1")),
            (ErrorPersistence.SEVERITY_WARNING, ValueError("Warning 2")),
            (ErrorPersistence.SEVERITY_ERROR, RuntimeError("Error 1")),
            (ErrorPersistence.SEVERITY_CRITICAL, MemoryError("Critical 1")),
        ]

        for severity, error in errors:
            error_persistence.log_error(error, severity=severity)

        # 获取统计信息
        stats = error_persistence.get_error_statistics(hours=24)

        assert stats['total_errors'] == 4
        assert stats['severity_breakdown']['WARNING'] == 2
        assert stats['severity_breakdown']['ERROR'] == 1
        assert stats['severity_breakdown']['CRITICAL'] == 1

    def test_error_type_distribution(self, error_persistence):
        """测试错误类型分布"""
        # 创建不同类型的错误
        for i in range(5):
            error_persistence.log_error(ValueError(f"Value error {i}"))

        for i in range(3):
            error_persistence.log_error(TypeError(f"Type error {i}"))

        for i in range(7):
            error_persistence.log_error(RuntimeError(f"Runtime error {i}"))

        stats = error_persistence.get_error_statistics(hours=24)

        assert stats['error_types']['ValueError'] == 5
        assert stats['error_types']['TypeError'] == 3
        assert stats['error_types']['RuntimeError'] == 7

    def test_top_patterns_in_statistics(self, error_persistence):
        """测试统计信息中的高频模式"""
        # 创建高频错误
        for i in range(15):
            error_persistence.log_error(ValueError("Most frequent error"))

        for i in range(8):
            error_persistence.log_error(TypeError("Second frequent error"))

        stats = error_persistence.get_error_statistics(hours=24)

        assert len(stats['top_patterns']) == 2
        assert stats['top_patterns'][0]['count'] == 15
        assert stats['top_patterns'][1]['count'] == 8


class TestErrorSearch:
    """测试错误搜索功能"""

    def test_search_by_keyword(self, error_persistence):
        """测试按关键词搜索"""
        # 创建不同消息的错误
        error_persistence.log_error(ValueError("CUDA out of memory"))
        error_persistence.log_error(RuntimeError("Invalid CUDA device"))
        error_persistence.log_error(TypeError("String expected"))

        # 搜索包含"CUDA"的错误
        results = error_persistence.search_errors(keyword="CUDA")

        assert len(results) == 2
        assert all('CUDA' in r['error_message'] for r in results)

    def test_search_by_severity(self, error_persistence):
        """测试按严重性搜索"""
        error_persistence.log_error(ValueError("Warning"), severity=ErrorPersistence.SEVERITY_WARNING)
        error_persistence.log_error(RuntimeError("Error"), severity=ErrorPersistence.SEVERITY_ERROR)
        error_persistence.log_error(MemoryError("Critical"), severity=ErrorPersistence.SEVERITY_CRITICAL)

        # 搜索ERROR级别
        results = error_persistence.search_errors(severity=ErrorPersistence.SEVERITY_ERROR)

        assert len(results) == 1
        assert results[0]['severity'] == 'ERROR'

    def test_search_by_error_type(self, error_persistence):
        """测试按错误类型搜索"""
        error_persistence.log_error(ValueError("Value error 1"))
        error_persistence.log_error(ValueError("Value error 2"))
        error_persistence.log_error(TypeError("Type error 1"))

        # 搜索ValueError类型
        results = error_persistence.search_errors(error_type="ValueError")

        assert len(results) == 2
        assert all(r['error_type'] == 'ValueError' for r in results)

    def test_search_limit(self, error_persistence):
        """测试搜索数量限制"""
        # 创建100个错误
        for i in range(100):
            error_persistence.log_error(ValueError(f"Error {i}"))

        # 限制返回10个
        results = error_persistence.search_errors(limit=10)

        assert len(results) == 10


class TestWebUIExport:
    """测试WebUI/API数据导出"""

    def test_export_for_webui_structure(self, error_persistence):
        """测试WebUI导出数据结构"""
        # 创建一些测试数据
        for i in range(10):
            error_persistence.log_error(ValueError(f"Test error {i}"))

        # 导出数据
        data = error_persistence.export_for_webui()

        # 验证数据结构
        assert 'statistics' in data
        assert 'patterns' in data
        assert 'recent_errors' in data
        assert 'timeline' in data
        assert 'generated_at' in data

        # 验证统计信息
        assert 'last_24h' in data['statistics']
        assert 'last_7d' in data['statistics']

        # 验证包含数据
        assert data['statistics']['last_24h']['total_errors'] == 10

    def test_export_to_json_file(self, error_persistence, temp_db):
        """测试导出到JSON文件"""
        # 创建测试数据
        error_persistence.log_error(ValueError("Test export"))

        # 导出到文件
        export_path = temp_db.parent / "export.json"
        data = error_persistence.export_for_webui(export_path=str(export_path))

        # 验证文件已创建
        assert export_path.exists()

        # 验证文件内容
        with open(export_path, 'r', encoding='utf-8') as f:
            file_data = json.load(f)

        assert file_data == data
        assert file_data['statistics']['last_24h']['total_errors'] > 0

    def test_timeline_aggregation(self, error_persistence):
        """测试时间线聚合"""
        # 创建错误
        for i in range(5):
            error_persistence.log_error(ValueError(f"Error {i}"))

        # 导出数据
        data = error_persistence.export_for_webui()

        # 验证时间线
        assert 'timeline' in data
        assert isinstance(data['timeline'], list)

        # 时间线应该包含按小时聚合的数据
        if len(data['timeline']) > 0:
            timeline_entry = data['timeline'][0]
            assert 'hour' in timeline_entry
            assert 'total' in timeline_entry
            assert 'by_severity' in timeline_entry


class TestErrorReport:
    """测试错误报告生成"""

    def test_generate_error_report(self, error_persistence):
        """测试生成Markdown报告"""
        # 创建测试数据
        for i in range(5):
            error_persistence.log_error(ValueError("Frequent error"))

        for i in range(2):
            error_persistence.log_error(TypeError("Less frequent error"))

        # 生成报告
        report = error_persistence.generate_error_report()

        # 验证报告内容
        assert "# 错误分析报告" in report
        assert "错误统计概览" in report
        assert "高频错误模式" in report
        assert "改进建议" in report

        # 验证包含错误信息
        assert "ValueError" in report
        assert "TypeError" in report

    def test_save_report_to_file(self, error_persistence, temp_db):
        """测试保存报告到文件"""
        # 创建测试数据
        error_persistence.log_error(ValueError("Test error"))

        # 生成并保存报告
        report_path = temp_db.parent / "error_report.md"
        report = error_persistence.generate_error_report(output_path=str(report_path))

        # 验证文件已创建
        assert report_path.exists()

        # 验证文件内容
        with open(report_path, 'r', encoding='utf-8') as f:
            file_content = f.read()

        assert file_content == report
        assert "# 错误分析报告" in file_content


class TestConvenienceFunctions:
    """测试便捷函数"""

    def test_log_training_error(self, temp_db):
        """测试log_training_error便捷函数"""
        # 使用全局单例需要覆盖数据库路径
        import apt.apt_model.utils.error_persistence as ep_module
        ep_module._global_error_persistence = ErrorPersistence(db_path=str(temp_db))

        try:
            raise ValueError("Training error")
        except ValueError as e:
            error_id = log_training_error(
                error=e,
                severity=ErrorPersistence.SEVERITY_ERROR,
                epoch=10,
                global_step=500,
                context={'batch_size': 16}
            )

        # 验证错误已记录
        assert error_id > 0

        # 验证上下文信息
        ep = get_global_error_persistence()
        errors = ep.search_errors(limit=1)

        assert len(errors) == 1
        assert errors[0]['epoch'] == 10
        assert errors[0]['global_step'] == 500
        assert errors[0]['context']['batch_size'] == 16

        ep.close()

    def test_global_error_persistence_singleton(self, temp_db):
        """测试全局单例模式"""
        import apt.apt_model.utils.error_persistence as ep_module

        # 重置全局实例
        if hasattr(ep_module, '_global_error_persistence'):
            delattr(ep_module, '_global_error_persistence')

        # 覆盖全局实例
        ep_module._global_error_persistence = ErrorPersistence(db_path=str(temp_db))

        # 获取两次应该是同一个实例
        ep1 = get_global_error_persistence()
        ep2 = get_global_error_persistence()

        assert ep1 is ep2

        ep1.close()


class TestContextManager:
    """测试上下文管理器"""

    def test_context_manager_usage(self, temp_db):
        """测试上下文管理器正确关闭连接"""
        with ErrorPersistence(db_path=str(temp_db)) as ep:
            ep.log_error(ValueError("Test error"))

            # 连接应该仍然打开
            assert ep.conn is not None

        # 退出上下文后，连接应该已关闭
        # 注意: sqlite3的Connection对象没有简单的方法检查是否关闭
        # 但我们可以尝试操作，应该会失败
        with pytest.raises(sqlite3.ProgrammingError):
            ep.cursor.execute("SELECT 1")


# ============================================================================
# 🔮 API Readiness Tests (未来API端点测试)
# ============================================================================

class TestAPIReadiness:
    """测试未来API端点的数据接口"""

    def test_api_endpoint_statistics(self, error_persistence):
        """
        测试统计信息API端点

        未来API: GET /api/errors/statistics
        """
        # 创建测试数据
        for i in range(10):
            error_persistence.log_error(ValueError(f"Error {i}"))

        # 模拟API响应
        stats = error_persistence.get_error_statistics(hours=24)

        # 验证API响应格式
        assert 'total_errors' in stats
        assert 'severity_breakdown' in stats
        assert 'error_types' in stats
        assert 'top_patterns' in stats
        assert stats['total_errors'] == 10

    def test_api_endpoint_patterns(self, error_persistence):
        """
        测试错误模式API端点

        未来API: GET /api/errors/patterns
        """
        # 创建测试数据
        for i in range(5):
            error_persistence.log_error(ValueError("Repeated error"))

        # 模拟API响应
        patterns = error_persistence.get_error_patterns(min_occurrences=1)

        # 验证API响应格式
        assert isinstance(patterns, list)
        assert len(patterns) > 0
        assert 'error_hash' in patterns[0]
        assert 'error_type' in patterns[0]
        assert 'occurrence_count' in patterns[0]

    def test_api_endpoint_resolve(self, error_persistence):
        """
        测试解决错误API端点

        未来API: POST /api/errors/resolve/{hash}
        """
        # 创建错误
        error_persistence.log_error(ValueError("Test error"))

        # 获取错误哈希
        patterns = error_persistence.get_error_patterns(min_occurrences=1, unresolved_only=False)
        error_hash = patterns[0]['error_hash']

        # 模拟API调用
        error_persistence.mark_pattern_resolved(error_hash)

        # 验证已解决
        patterns_updated = error_persistence.get_error_patterns(min_occurrences=1, unresolved_only=False)
        assert patterns_updated[0]['resolved'] == True

    def test_api_endpoint_search(self, error_persistence):
        """
        测试搜索API端点

        未来API: GET /api/errors/search?keyword=xxx
        """
        # 创建测试数据
        error_persistence.log_error(ValueError("CUDA memory error"))
        error_persistence.log_error(RuntimeError("File not found"))

        # 模拟API调用
        results = error_persistence.search_errors(keyword="CUDA", limit=10)

        # 验证API响应
        assert isinstance(results, list)
        assert len(results) == 1
        assert 'error_message' in results[0]
        assert 'CUDA' in results[0]['error_message']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
