import logging
import logging.handlers
import os
from pathlib import Path
from typing import Optional, Union
import sys
import time
from datetime import datetime
import json
import yaml
import traceback


class ColoredFormatter(logging.Formatter):
    """带颜色的日志格式化器"""

    # 颜色映射
    COLORS = {
        'DEBUG': '\033[94m',  # 蓝色
        'INFO': '\033[92m',  # 绿色
        'WARNING': '\033[93m',  # 黄色
        'ERROR': '\033[91m',  # 红色
        'CRITICAL': '\033[91m',  # 红色
        'RESET': '\033[0m'  # 重置
    }

    def format(self, record: logging.LogRecord) -> str:
        """格式化日志记录"""
        # 给日志级别添加颜色
        level_name = record.levelname
        if level_name in self.COLORS:
            colored_level_name = f"{self.COLORS[level_name]}{level_name}{self.COLORS['RESET']}"
            record.levelname = colored_level_name

        return super().format(record)


class JsonFormatter(logging.Formatter):
    """JSON格式的日志格式化器"""

    def format(self, record: logging.LogRecord) -> str:
        """格式化为JSON字符串"""
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }

        # 添加异常信息
        if record.exc_info:
            log_data['exception'] = {
                'type': str(record.exc_info[0].__name__),
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }

        # 添加额外字段
        if hasattr(record, 'extras'):
            log_data.update(record.extras)

        return json.dumps(log_data)


class LogManager:
    """日志管理器"""

    def __init__(self,
                 name: str,
                 log_dir: Union[str, Path],
                 level: str = 'INFO',
                 backup_count: int = 30,
                 console_output: bool = True,
                 json_output: bool = False):
        """
        初始化日志管理器
        Args:
            name: 日志器名称
            log_dir: 日志保存目录
            level: 日志级别
            backup_count: 保留的日志文件数量
            console_output: 是否输出到控制台
            json_output: 是否使用JSON格式
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.level = getattr(logging, level.upper())
        self.backup_count = backup_count
        self.console_output = console_output
        self.json_output = json_output

        # 创建日志目录
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 获取日志器
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.level)

        # 清除现有的处理器
        self.logger.handlers = []

        # 添加处理器
        self._add_handlers()

    def _add_handlers(self):
        """添加日志处理器"""
        handlers = []

        # 文件处理器
        log_file = self.log_dir / f'{self.name}.log'
        file_handler = logging.handlers.TimedRotatingFileHandler(
            filename=log_file,
            when='midnight',
            interval=1,
            backupCount=self.backup_count,
            encoding='utf-8'
        )

        # 根据配置选择格式化器
        if self.json_output:
            file_formatter = JsonFormatter()
        else:
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )

        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)

        # 控制台处理器
        if self.console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_formatter = ColoredFormatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            handlers.append(console_handler)

        # 添加所有处理器
        for handler in handlers:
            self.logger.addHandler(handler)

    def get_logger(self) -> logging.Logger:
        """获取日志器"""
        return self.logger


class ExperimentLogger:
    """实验日志器"""

    def __init__(self,
                 experiment_name: str,
                 base_dir: Union[str, Path],
                 config: Optional[dict] = None):
        """
        初始化实验日志器
        Args:
            experiment_name: 实验名称
            base_dir: 基础目录
            config: 实验配置
        """
        self.experiment_name = experiment_name
        self.base_dir = Path(base_dir)
        self.config = config

        # 创建实验目录
        self.experiment_dir = self._create_experiment_dir()

        # 初始化日志器
        self.log_manager = LogManager(
            name=experiment_name,
            log_dir=self.experiment_dir / 'logs'
        )
        self.logger = self.log_manager.get_logger()

        # 保存配置
        if config:
            self._save_config()

    def _create_experiment_dir(self) -> Path:
        """创建实验目录"""
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        experiment_dir = self.base_dir / f'{self.experiment_name}_{timestamp}'
        experiment_dir.mkdir(parents=True, exist_ok=True)

        # 创建子目录
        (experiment_dir / 'logs').mkdir(exist_ok=True)
        (experiment_dir / 'checkpoints').mkdir(exist_ok=True)
        (experiment_dir / 'results').mkdir(exist_ok=True)

        return experiment_dir

    def _save_config(self):
        """保存配置文件"""
        config_file = self.experiment_dir / 'config.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)

    def log_metrics(self, metrics: dict, step: Optional[int] = None):
        """
        记录指标
        Args:
            metrics: 指标字典
            step: 当前步数
        """
        prefix = f"Step {step} - " if step is not None else ""
        metrics_str = " - ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"{prefix}{metrics_str}")

    def log_artifact(self,
                     name: str,
                     artifact: Union[str, Path, bytes],
                     artifact_type: str = 'file'):
        """
        记录制品
        Args:
            name: 制品名称
            artifact: 制品内容
            artifact_type: 制品类型
        """
        artifact_dir = self.experiment_dir / 'artifacts'
        artifact_dir.mkdir(exist_ok=True)

        if artifact_type == 'file':
            if isinstance(artifact, (str, Path)):
                # 复制文件
                import shutil
                shutil.copy2(artifact, artifact_dir / name)
            else:
                # 写入二进制数据
                with open(artifact_dir / name, 'wb') as f:
                    f.write(artifact)

        self.logger.info(f"保存制品: {name}")

    def log_figure(self, name: str, figure):
        """
        保存matplotlib图表
        Args:
            name: 图表名称
            figure: matplotlib图表对象
        """
        figure_dir = self.experiment_dir / 'figures'
        figure_dir.mkdir(exist_ok=True)
        figure.savefig(figure_dir / f'{name}.png')
        self.logger.info(f"保存图表: {name}")


def setup_logger(name: str,
                 log_dir: Optional[Union[str, Path]] = None,
                 level: str = 'INFO') -> logging.Logger:
    """
    快速设置日志器
    Args:
        name: 日志器名称
        log_dir: 日志目录
        level: 日志级别
    Returns:
        配置好的日志器
    """
    if log_dir is None:
        log_dir = Path.cwd() / 'logs'

    log_manager = LogManager(
        name=name,
        log_dir=log_dir,
        level=level
    )
    return log_manager.get_logger()