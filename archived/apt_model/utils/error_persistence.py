#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
错误持久化系统

提供错误日志持久化、模式分析和查询功能。
支持SQLite数据库存储，错误模式识别，以及未来WebUI/API集成。
"""

import os
import re
import sqlite3
import traceback
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from collections import defaultdict, Counter


class ErrorPersistence:
    """
    错误持久化管理器

    功能:
    - SQLite数据库存储错误日志
    - 错误分类和严重性评估
    - 错误模式识别和聚合
    - 错误统计和趋势分析
    - 未来WebUI/API数据导出
    """

    # 错误严重性级别
    SEVERITY_DEBUG = 0
    SEVERITY_INFO = 1
    SEVERITY_WARNING = 2
    SEVERITY_ERROR = 3
    SEVERITY_CRITICAL = 4

    SEVERITY_NAMES = {
        0: 'DEBUG',
        1: 'INFO',
        2: 'WARNING',
        3: 'ERROR',
        4: 'CRITICAL'
    }

    def __init__(self, db_path: str = ".cache/errors/errors.db", auto_init: bool = True):
        """
        初始化错误持久化系统

        参数:
            db_path: SQLite数据库路径（相对路径，可迁移）
            auto_init: 是否自动初始化数据库
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = None
        self.cursor = None

        if auto_init:
            self._init_database()

    def _init_database(self):
        """初始化SQLite数据库和表结构"""
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.cursor = self.conn.cursor()

        # 创建错误日志表
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS error_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                severity INTEGER NOT NULL,
                module TEXT,
                function TEXT,
                error_type TEXT,
                error_message TEXT,
                error_hash TEXT,
                stacktrace TEXT,
                context TEXT,
                epoch INTEGER,
                global_step INTEGER,
                resolved BOOLEAN DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 创建错误模式表（用于聚合相同错误）
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS error_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                error_hash TEXT UNIQUE NOT NULL,
                error_type TEXT,
                error_pattern TEXT,
                occurrence_count INTEGER DEFAULT 1,
                first_seen DATETIME,
                last_seen DATETIME,
                severity_max INTEGER,
                resolved BOOLEAN DEFAULT 0
            )
        """)

        # 创建索引以加速查询
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_error_hash
            ON error_logs(error_hash)
        """)

        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp
            ON error_logs(timestamp)
        """)

        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_severity
            ON error_logs(severity)
        """)

        self.conn.commit()

    def log_error(
        self,
        error: Exception,
        severity: int = SEVERITY_ERROR,
        module: str = None,
        function: str = None,
        context: Dict[str, Any] = None,
        epoch: int = None,
        global_step: int = None
    ) -> int:
        """
        记录错误到数据库

        参数:
            error: 异常对象
            severity: 错误严重性级别
            module: 模块名称
            function: 函数名称
            context: 上下文信息（字典）
            epoch: 训练轮次
            global_step: 全局训练步数

        返回:
            error_id: 错误记录ID
        """
        timestamp = datetime.now().isoformat()
        error_type = type(error).__name__
        error_message = str(error)
        stacktrace = traceback.format_exc()

        # 生成错误哈希（用于模式识别）
        error_hash = self._generate_error_hash(error_type, error_message, stacktrace)

        # 序列化上下文
        context_json = json.dumps(context) if context else None

        # 插入错误日志
        self.cursor.execute("""
            INSERT INTO error_logs
            (timestamp, severity, module, function, error_type, error_message,
             error_hash, stacktrace, context, epoch, global_step)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp, severity, module, function, error_type, error_message,
            error_hash, stacktrace, context_json, epoch, global_step
        ))

        error_id = self.cursor.lastrowid
        self.conn.commit()

        # 更新错误模式
        self._update_error_pattern(error_hash, error_type, error_message, severity, timestamp)

        return error_id

    def _generate_error_hash(self, error_type: str, error_message: str, stacktrace: str) -> str:
        """
        生成错误哈希（用于识别相同错误）

        通过提取错误类型、消息模式和调用栈关键部分生成哈希
        """
        # 提取错误消息的模式（去除数字、路径等动态部分）
        message_pattern = re.sub(r'\d+', 'N', error_message)
        message_pattern = re.sub(r'/[^\s]+', '/PATH', message_pattern)
        message_pattern = re.sub(r'0x[0-9a-fA-F]+', '0xADDR', message_pattern)

        # 提取调用栈关键部分（只保留文件名和函数名）
        stack_lines = stacktrace.split('\n')
        stack_pattern = []
        for line in stack_lines:
            if 'File' in line and 'line' in line:
                # 提取文件名和函数
                match = re.search(r'File "([^"]+)", line \d+, in (\w+)', line)
                if match:
                    filename = Path(match.group(1)).name
                    function = match.group(2)
                    stack_pattern.append(f"{filename}:{function}")

        # 组合哈希内容
        hash_content = f"{error_type}|{message_pattern}|{'->'.join(stack_pattern[-3:])}"

        # 生成MD5哈希
        return hashlib.md5(hash_content.encode()).hexdigest()[:16]

    def _update_error_pattern(
        self,
        error_hash: str,
        error_type: str,
        error_message: str,
        severity: int,
        timestamp: str
    ):
        """更新错误模式表"""
        # 检查是否已存在
        self.cursor.execute("""
            SELECT id, occurrence_count, severity_max, first_seen
            FROM error_patterns
            WHERE error_hash = ?
        """, (error_hash,))

        result = self.cursor.fetchone()

        # 提取错误模式（泛化的错误消息）
        error_pattern = re.sub(r'\d+', 'N', error_message)
        error_pattern = re.sub(r'/[^\s]+', '/PATH', error_pattern)

        if result:
            # 更新现有模式
            pattern_id, count, max_severity, first_seen = result
            new_count = count + 1
            new_max_severity = max(max_severity, severity)

            self.cursor.execute("""
                UPDATE error_patterns
                SET occurrence_count = ?,
                    severity_max = ?,
                    last_seen = ?,
                    error_pattern = ?
                WHERE id = ?
            """, (new_count, new_max_severity, timestamp, error_pattern, pattern_id))
        else:
            # 创建新模式
            self.cursor.execute("""
                INSERT INTO error_patterns
                (error_hash, error_type, error_pattern, occurrence_count,
                 first_seen, last_seen, severity_max)
                VALUES (?, ?, ?, 1, ?, ?, ?)
            """, (error_hash, error_type, error_pattern, timestamp, timestamp, severity))

        self.conn.commit()

    def get_error_patterns(
        self,
        min_occurrences: int = 2,
        unresolved_only: bool = True,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        获取错误模式列表

        参数:
            min_occurrences: 最小出现次数
            unresolved_only: 只显示未解决的错误
            limit: 返回数量限制

        返回:
            错误模式列表
        """
        query = """
            SELECT error_hash, error_type, error_pattern, occurrence_count,
                   first_seen, last_seen, severity_max, resolved
            FROM error_patterns
            WHERE occurrence_count >= ?
        """
        params = [min_occurrences]

        if unresolved_only:
            query += " AND resolved = 0"

        query += " ORDER BY occurrence_count DESC, last_seen DESC LIMIT ?"
        params.append(limit)

        self.cursor.execute(query, params)

        patterns = []
        for row in self.cursor.fetchall():
            patterns.append({
                'error_hash': row[0],
                'error_type': row[1],
                'error_pattern': row[2],
                'occurrence_count': row[3],
                'first_seen': row[4],
                'last_seen': row[5],
                'severity': self.SEVERITY_NAMES.get(row[6], 'UNKNOWN'),
                'resolved': bool(row[7])
            })

        return patterns

    def get_error_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """
        获取错误统计信息

        参数:
            hours: 统计最近N小时的错误

        返回:
            统计信息字典
        """
        from datetime import timedelta
        cutoff_time = (datetime.now() - timedelta(hours=hours)).isoformat()

        # 总错误数
        self.cursor.execute("""
            SELECT COUNT(*) FROM error_logs WHERE timestamp >= ?
        """, (cutoff_time,))
        total_errors = self.cursor.fetchone()[0]

        # 按严重性分组
        self.cursor.execute("""
            SELECT severity, COUNT(*)
            FROM error_logs
            WHERE timestamp >= ?
            GROUP BY severity
        """, (cutoff_time,))

        severity_counts = {}
        for row in self.cursor.fetchall():
            severity_counts[self.SEVERITY_NAMES.get(row[0], 'UNKNOWN')] = row[1]

        # 按错误类型分组
        self.cursor.execute("""
            SELECT error_type, COUNT(*)
            FROM error_logs
            WHERE timestamp >= ?
            GROUP BY error_type
            ORDER BY COUNT(*) DESC
            LIMIT 10
        """, (cutoff_time,))

        error_type_counts = {row[0]: row[1] for row in self.cursor.fetchall()}

        # 最常见的错误模式
        self.cursor.execute("""
            SELECT error_pattern, occurrence_count, severity_max
            FROM error_patterns
            WHERE last_seen >= ?
            ORDER BY occurrence_count DESC
            LIMIT 5
        """, (cutoff_time,))

        top_patterns = []
        for row in self.cursor.fetchall():
            top_patterns.append({
                'pattern': row[0],
                'count': row[1],
                'severity': self.SEVERITY_NAMES.get(row[2], 'UNKNOWN')
            })

        return {
            'total_errors': total_errors,
            'severity_breakdown': severity_counts,
            'error_types': error_type_counts,
            'top_patterns': top_patterns,
            'time_window_hours': hours
        }

    def mark_pattern_resolved(self, error_hash: str):
        """标记错误模式为已解决"""
        self.cursor.execute("""
            UPDATE error_patterns SET resolved = 1 WHERE error_hash = ?
        """, (error_hash,))

        self.cursor.execute("""
            UPDATE error_logs SET resolved = 1 WHERE error_hash = ?
        """, (error_hash,))

        self.conn.commit()

    def search_errors(
        self,
        keyword: str = None,
        severity: int = None,
        error_type: str = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        搜索错误日志

        参数:
            keyword: 关键词（搜索错误消息）
            severity: 严重性级别
            error_type: 错误类型
            limit: 返回数量限制

        返回:
            错误日志列表
        """
        query = "SELECT * FROM error_logs WHERE 1=1"
        params = []

        if keyword:
            query += " AND error_message LIKE ?"
            params.append(f"%{keyword}%")

        if severity is not None:
            query += " AND severity = ?"
            params.append(severity)

        if error_type:
            query += " AND error_type = ?"
            params.append(error_type)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        self.cursor.execute(query, params)

        errors = []
        for row in self.cursor.fetchall():
            errors.append({
                'id': row[0],
                'timestamp': row[1],
                'severity': self.SEVERITY_NAMES.get(row[2], 'UNKNOWN'),
                'module': row[3],
                'function': row[4],
                'error_type': row[5],
                'error_message': row[6],
                'error_hash': row[7],
                'stacktrace': row[8],
                'context': json.loads(row[9]) if row[9] else None,
                'epoch': row[10],
                'global_step': row[11],
                'resolved': bool(row[12])
            })

        return errors

    # ========================================================================
    # 🔮 WebUI/API Export Interface
    # ========================================================================

    def export_for_webui(self, export_path: str = None) -> Dict[str, Any]:
        """
        导出错误数据供WebUI/API使用

        未来API端点:
        - GET /api/errors/statistics
        - GET /api/errors/patterns
        - GET /api/errors/timeline
        - POST /api/errors/resolve/{hash}

        参数:
            export_path: JSON文件导出路径（可选）

        返回:
            完整的错误分析数据
        """
        # 获取统计信息
        stats_24h = self.get_error_statistics(hours=24)
        stats_7d = self.get_error_statistics(hours=24 * 7)

        # 获取错误模式
        patterns = self.get_error_patterns(min_occurrences=1, unresolved_only=False, limit=100)

        # 获取最近错误
        recent_errors = self.search_errors(limit=50)

        # 错误时间线（按小时聚合）
        self.cursor.execute("""
            SELECT
                strftime('%Y-%m-%d %H:00:00', timestamp) as hour,
                severity,
                COUNT(*) as count
            FROM error_logs
            WHERE timestamp >= datetime('now', '-7 days')
            GROUP BY hour, severity
            ORDER BY hour DESC
        """)

        timeline = defaultdict(lambda: {'total': 0, 'by_severity': {}})
        for row in self.cursor.fetchall():
            hour, severity, count = row
            severity_name = self.SEVERITY_NAMES.get(severity, 'UNKNOWN')
            timeline[hour]['total'] += count
            timeline[hour]['by_severity'][severity_name] = count

        timeline_list = [
            {'hour': hour, **data}
            for hour, data in sorted(timeline.items(), reverse=True)
        ]

        data = {
            'statistics': {
                'last_24h': stats_24h,
                'last_7d': stats_7d
            },
            'patterns': patterns,
            'recent_errors': recent_errors,
            'timeline': timeline_list,
            'generated_at': datetime.now().isoformat()
        }

        # 导出到JSON文件
        if export_path:
            export_file = Path(export_path)
            export_file.parent.mkdir(parents=True, exist_ok=True)
            with open(export_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        return data

    def generate_error_report(self, output_path: str = None) -> str:
        """
        生成错误分析报告（Markdown格式）

        参数:
            output_path: 报告保存路径（可选）

        返回:
            Markdown格式的报告内容
        """
        stats = self.get_error_statistics(hours=24)
        patterns = self.get_error_patterns(min_occurrences=2, limit=20)

        report = []
        report.append("# 错误分析报告")
        report.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"\n统计时间窗口: 最近 {stats['time_window_hours']} 小时\n")

        # 统计概览
        report.append("## 错误统计概览\n")
        report.append(f"- **总错误数**: {stats['total_errors']}\n")

        report.append("### 按严重性分类\n")
        for severity, count in stats['severity_breakdown'].items():
            report.append(f"- **{severity}**: {count}")

        report.append("\n### 错误类型分布\n")
        for error_type, count in stats['error_types'].items():
            report.append(f"- `{error_type}`: {count}")

        # 高频错误模式
        report.append("\n## 高频错误模式\n")
        if patterns:
            report.append("| 错误类型 | 错误模式 | 出现次数 | 严重性 | 状态 |")
            report.append("|----------|----------|----------|--------|------|")
            for p in patterns:
                status = "✅ 已解决" if p['resolved'] else "⚠️ 未解决"
                report.append(
                    f"| {p['error_type']} | {p['error_pattern'][:50]}... | "
                    f"{p['occurrence_count']} | {p['severity']} | {status} |"
                )
        else:
            report.append("*无重复错误模式*")

        # 最常见错误
        report.append("\n## 最常见错误 (Top 5)\n")
        if stats['top_patterns']:
            for i, pattern in enumerate(stats['top_patterns'], 1):
                report.append(f"{i}. **[{pattern['severity']}]** {pattern['pattern']} "
                            f"(出现 {pattern['count']} 次)")
        else:
            report.append("*暂无数据*")

        # 建议
        report.append("\n## 改进建议\n")
        if stats['total_errors'] > 100:
            report.append("- ⚠️ 错误数量较高，建议检查训练配置和数据质量")

        critical_count = stats['severity_breakdown'].get('CRITICAL', 0)
        if critical_count > 0:
            report.append(f"- 🚨 发现 {critical_count} 个严重错误，需要立即处理")

        if len(patterns) > 10:
            report.append(f"- 🔍 发现 {len(patterns)} 个重复错误模式，建议进行模式分析和批量修复")

        report_text = '\n'.join(report)

        # 保存报告
        if output_path:
            report_file = Path(output_path)
            report_file.parent.mkdir(parents=True, exist_ok=True)
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report_text)

        return report_text

    def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()

    def __enter__(self):
        """支持上下文管理器"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文时自动关闭"""
        self.close()


# ============================================================================
# 便捷函数
# ============================================================================

def get_global_error_persistence() -> ErrorPersistence:
    """
    获取全局错误持久化实例（单例模式）

    返回:
        ErrorPersistence实例
    """
    global _global_error_persistence

    if '_global_error_persistence' not in globals():
        _global_error_persistence = ErrorPersistence()

    return _global_error_persistence


def log_training_error(
    error: Exception,
    severity: int = ErrorPersistence.SEVERITY_ERROR,
    epoch: int = None,
    global_step: int = None,
    context: Dict[str, Any] = None
):
    """
    便捷函数: 记录训练过程中的错误

    参数:
        error: 异常对象
        severity: 严重性级别
        epoch: 训练轮次
        global_step: 全局步数
        context: 额外上下文信息
    """
    ep = get_global_error_persistence()

    # 自动提取调用栈信息
    import inspect
    frame = inspect.currentframe().f_back
    module = frame.f_globals.get('__name__', 'unknown')
    function = frame.f_code.co_name

    return ep.log_error(
        error=error,
        severity=severity,
        module=module,
        function=function,
        context=context,
        epoch=epoch,
        global_step=global_step
    )
