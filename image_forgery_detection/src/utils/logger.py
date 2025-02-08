from __future__ import annotations

import logging
import logging.handlers
import os
from pathlib import Path
from typing import Optional, Union, Dict, Any, List
import sys
import time
from datetime import datetime
import json
import yaml
import traceback
import threading
from dataclasses import dataclass, field
import queue
import socket
import weakref
import atexit
from concurrent.futures import ThreadPoolExecutor

# 类型别名定义
PathLike = Union[str, Path]
LogLevel = Union[str, int]


@dataclass
class LogConfig:
    """日志配置类"""
    name: str
    log_dir: PathLike
    level: LogLevel = 'INFO'
    backup_count: int = 30
    console_output: bool = True
    json_output: bool = False
    async_logging: bool = True
    queue_size: int = 1000
    max_bytes: int = 100 * 1024 * 1024  # 100MB
    encoding: str = 'utf-8'
    errors: str = 'replace'

    def __post_init__(self):
        """验证和处理配置"""
        # 转换日志级别
        if isinstance(self.level, str):
            self.level = getattr(logging, self.level.upper())

        # 验证配置
        if self.max_bytes < 0:
            raise ValueError("max_bytes must be non-negative")
        if self.backup_count < 0:
            raise ValueError("backup_count must be non-negative")
        if self.queue_size < 1:
            raise ValueError("queue_size must be positive")

        # 转换路径
        self.log_dir = Path(self.log_dir)


class MetricsCollector:
    """指标收集器"""

    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}
        self._lock = threading.Lock()

    def add_metric(self, name: str, value: float):
        """添加指标值"""
        with self._lock:
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append(value)

    def get_metrics(self, name: str) -> List[float]:
        """获取指标历史值"""
        with self._lock:
            return self.metrics.get(name, []).copy()

    def clear(self):
        """清空指标"""
        with self._lock:
            self.metrics.clear()


class AsyncLogQueue:
    """异步日志队列"""

    def __init__(self, queue_size: int = 1000):
        self.queue = queue.Queue(maxsize=queue_size)
        self.running = True
        self.worker = threading.Thread(target=self._process_logs, daemon=True)
        self.handlers: List[logging.Handler] = []
        self.worker.start()

        # 注册清理函数
        atexit.register(self.stop)

    def add_record(self, record: logging.LogRecord):
        """添加日志记录"""
        try:
            self.queue.put_nowait(record)
        except queue.Full:
            # 队列满时直接处理
            self._handle_record(record)

    def add_handler(self, handler: logging.Handler):
        """添加处理器"""
        self.handlers.append(handler)

    def _process_logs(self):
        """处理日志队列"""
        while self.running or not self.queue.empty():
            try:
                record = self.queue.get(timeout=0.1)
                self._handle_record(record)
                self.queue.task_done()
            except queue.Empty:
                continue
            except Exception:
                sys.stderr.write('Error in async logger\n')
                traceback.print_exc(file=sys.stderr)

    def _handle_record(self, record: logging.LogRecord):
        """处理单条日志记录"""
        for handler in self.handlers:
            if record.levelno >= handler.level:
                handler.handle(record)

    def stop(self):
        """停止处理"""
        self.running = False
        if self.worker.is_alive():
            self.worker.join(timeout=1.0)
        # 处理剩余日志
        while not self.queue.empty():
            try:
                record = self.queue.get_nowait()
                self._handle_record(record)
            except queue.Empty:
                break


class LogContext:
    """日志上下文管理器"""

    def __init__(self):
        self._context = threading.local()

    def get(self) -> Dict[str, Any]:
        """获取当前上下文"""
        if not hasattr(self._context, 'data'):
            self._context.data = {}
        return self._context.data

    def set(self, **kwargs):
        """设置上下文数据"""
        self.get().update(kwargs)

    def clear(self):
        """清空上下文"""
        if hasattr(self._context, 'data'):
            self._context.data.clear()


class EnhancedColoredFormatter(logging.Formatter):
    """增强的彩色日志格式化器"""

    # 使用256色定义更丰富的颜色
    COLORS = {
        'DEBUG': '\033[38;5;105m',  # 淡紫色
        'INFO': '\033[38;5;82m',  # 亮绿色
        'WARNING': '\033[38;5;214m',  # 橙色
        'ERROR': '\033[38;5;196m',  # 亮红色
        'CRITICAL': '\033[48;5;196;38;5;231m',  # 红底白字
        'RESET': '\033[0m'
    }

    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None):
        super().__init__(fmt, datefmt)
        # 使用线程局部存储缓存格式化器
        self._thread_data = threading.local()

    def format(self, record: logging.LogRecord) -> str:
        """格式化日志记录"""
        # 初始化线程局部缓存
        if not hasattr(self._thread_data, 'formatters'):
            self._thread_data.formatters = {}

        # 为每个日志级别创建专用格式化器
        if record.levelname not in self._thread_data.formatters:
            color = self.COLORS.get(record.levelname, '')
            fmt = self._fmt.replace(
                '%(levelname)s',
                f'{color}%(levelname)s{self.COLORS["RESET"]}'
            )
            self._thread_data.formatters[record.levelname] = logging.Formatter(fmt, self.datefmt)

        # 使用缓存的格式化器
        return self._thread_data.formatters[record.levelname].format(record)


class EnhancedJsonFormatter(logging.Formatter):
    """增强的JSON日志格式化器"""

    def __init__(self, **kwargs):
        super().__init__()
        self.default_fields = {
            'hostname': socket.gethostname(),
            'app_name': kwargs.get('app_name', ''),
            'environment': kwargs.get('environment', ''),
        }

    def format(self, record: logging.LogRecord) -> str:
        """格式化为JSON字符串"""
        try:
            # 基础日志信息
            log_data = {
                'timestamp': datetime.fromtimestamp(record.created).isoformat(),
                'level': record.levelname,
                'logger': record.name,
                'message': record.getMessage(),
                'module': record.module,
                'function': record.funcName,
                'line': record.lineno,
                'thread': {
                    'id': record.thread,
                    'name': record.threadName
                },
                'process': {
                    'id': record.process,
                    'name': record.processName
                }
            }

            # 添加默认字段
            log_data.update(self.default_fields)

            # 添加上下文数据
            context = getattr(record, 'context', {})
            if context:
                log_data['context'] = context

            # 添加异常信息
            if record.exc_info:
                log_data['exception'] = {
                    'type': record.exc_info[0].__name__,
                    'message': str(record.exc_info[1]),
                    'traceback': self.formatException(record.exc_info)
                }

            # 添加自定义属性
            if hasattr(record, 'extra_fields'):
                log_data.update(record.extra_fields)

            return json.dumps(log_data, default=str)

        except Exception as e:
            # 异常时返回基础错误信息
            return json.dumps({
                'error': 'Failed to format log record',
                'error_message': str(e),
                'original_message': record.getMessage()
            })


class CompressedRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """支持压缩的文件轮转处理器"""

    def __init__(self, filename: str, mode: str = 'a', maxBytes: int = 0,
                 backupCount: int = 0, encoding: Optional[str] = None,
                 delay: bool = False, errors: Optional[str] = None):
        super().__init__(filename, mode, maxBytes, backupCount, encoding, delay, errors)
        self.rotator = self._rotator
        self.namer = self._namer

    def _rotator(self, source: str, dest: str) -> None:
        """执行日志文件压缩"""
        import gzip
        try:
            with open(source, 'rb') as f_in:
                with gzip.open(f"{dest}.gz", 'wb') as f_out:
                    f_out.writelines(f_in)
            os.remove(source)
        except Exception:
            # 压缩失败时执行普通轮转
            if os.path.exists(source):
                os.rename(source, dest)

    def _namer(self, default_name: str) -> str:
        """生成轮转文件名"""
        # 使用时间戳作为文件名后缀
        return f"{default_name}.{time.strftime('%Y%m%d_%H%M%S')}"

    def rotation_filename(self, default_name: str) -> str:
        """重写轮转文件名生成方法"""
        return self.namer(default_name)

    def do_rollover(self) -> None:
        """执行日志轮转"""
        if self.stream:
            self.stream.close()
            self.stream = None

        # 处理备份文件
        if self.backupCount > 0:
            # 获取所有备份文件
            dir_name, base_name = os.path.split(self.baseFilename)
            backup_files = []
            for f in os.listdir(dir_name):
                if f.startswith(base_name) and f != base_name:
                    backup_files.append(os.path.join(dir_name, f))

            # 删除多余的备份
            if len(backup_files) >= self.backupCount:
                backup_files.sort()
                excess = len(backup_files) - self.backupCount + 1
                for f in backup_files[:excess]:
                    try:
                        os.remove(f)
                    except Exception:
                        pass

        # 执行轮转
        dfn = self.rotation_filename(self.baseFilename)
        self.rotator(self.baseFilename, dfn)

        if not self.delay:
            self.stream = self._open()


class AsyncHandler(logging.Handler):
    """异步日志处理器"""

    def __init__(self, queue_size: int = 1000):
        super().__init__()
        self.queue = queue.Queue(maxsize=queue_size)
        self.handlers: List[logging.Handler] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._process_logs)
        self._thread.daemon = True
        self._thread.start()

    def emit(self, record: logging.LogRecord) -> None:
        """发送日志记录"""
        try:
            self.queue.put_nowait(record)
        except queue.Full:
            # 队列满时直接处理
            self._handle_record(record)

    def _process_logs(self) -> None:
        """处理日志队列"""
        while not self._stop.is_set() or not self.queue.empty():
            try:
                record = self.queue.get(timeout=0.1)
                self._handle_record(record)
                self.queue.task_done()
            except queue.Empty:
                continue
            except Exception:
                import sys
                sys.stderr.write('Error in async handler\n')
                traceback.print_exc(file=sys.stderr)

    def _handle_record(self, record: logging.LogRecord) -> None:
        """处理单条日志记录"""
        for handler in self.handlers:
            if record.levelno >= handler.level:
                handler.handle(record)

    def addHandler(self, handler: logging.Handler) -> None:
        """添加处理器"""
        self.handlers.append(handler)

    def close(self) -> None:
        """关闭处理器"""
        self._stop.set()
        self._thread.join(timeout=2.0)
        # 处理剩余日志
        while not self.queue.empty():
            try:
                record = self.queue.get_nowait()
                self._handle_record(record)
            except queue.Empty:
                break
        for handler in self.handlers:
            handler.close()
        super().close()


class LogManager:
    """日志管理器,实现单例模式"""

    _instances = weakref.WeakValueDictionary()
    _lock = threading.Lock()

    def __new__(cls, name: str, *args, **kwargs) -> 'LogManager':
        """确保每个名称只创建一个实例"""
        with cls._lock:
            if name not in cls._instances:
                instance = super().__new__(cls)
                cls._instances[name] = instance
            return cls._instances[name]

    def __init__(self, name: str, config: Optional[LogConfig] = None):
        """初始化日志管理器"""
        # 避免重复初始化
        if hasattr(self, '_initialized'):
            return

        self.config = config or LogConfig(name=name, log_dir='logs')
        self.name = name
        self.log_dir = Path(self.config.log_dir)
        self.level = self._parse_level(self.config.level)
        self.context = LogContext()

        # 创建日志目录
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 初始化日志器
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.level)

        # 清除现有处理器
        self.logger.handlers = []

        # 创建异步处理器
        if self.config.async_logging:
            self.async_handler = AsyncHandler(self.config.queue_size)
            self.logger.addHandler(self.async_handler)

        # 添加处理器
        self._setup_handlers()

        self._initialized = True

    def _parse_level(self, level: LogLevel) -> int:
        """解析日志级别"""
        if isinstance(level, str):
            return getattr(logging, level.upper())
        return level

    def _setup_handlers(self) -> None:
        """设置日志处理器"""
        handlers = []

        # 文件处理器
        log_file = self.log_dir / f'{self.name}.log'
        file_handler = CompressedRotatingFileHandler(
            filename=str(log_file),
            maxBytes=self.config.max_bytes,
            backupCount=self.config.backup_count,
            encoding=self.config.encoding,
            errors=self.config.errors
        )

        # 选择格式化器
        if self.config.json_output:
            file_formatter = EnhancedJsonFormatter(
                app_name=self.name,
                environment=os.getenv('ENV', 'development')
            )
        else:
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )

        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)

        # 控制台处理器
        if self.config.console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_formatter = EnhancedColoredFormatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            handlers.append(console_handler)

        # 添加处理器
        if self.config.async_logging:
            for handler in handlers:
                self.async_handler.addHandler(handler)
        else:
            for handler in handlers:
                self.logger.addHandler(handler)

    def get_logger(self) -> logging.Logger:
        """获取日志器"""
        return self.logger

    @contextmanager
    def log_context(self, **context):
        """日志上下文管理器"""
        original = self.context.get()
        self.context.set(**context)
        try:
            yield
        finally:
            self.context.clear()
            self.context.set(**original)

    def cleanup(self):
        """清理资源"""
        if hasattr(self, 'async_handler'):
            self.async_handler.close()


class ExperimentLogger:
    """实验日志器,用于记录实验过程和结果"""

    def __init__(self, experiment_name: str, base_dir: PathLike,
                 config: Optional[Dict[str, Any]] = None):
        """初始化实验日志器"""
        self.experiment_name = experiment_name
        self.base_dir = Path(base_dir)
        self.config = config
        self.metrics_collector = MetricsCollector()

        # 创建实验目录
        self.experiment_dir = self._create_experiment_dir()

        # 初始化日志器
        log_config = LogConfig(
            name=experiment_name,
            log_dir=self.experiment_dir / 'logs',
            json_output=True,
            console_output=True,
            async_logging=True
        )
        self.log_manager = LogManager(experiment_name, log_config)
        self.logger = self.log_manager.get_logger()

        # 保存配置
        if config:
            self._save_config()

        # 线程池
        self._executor = ThreadPoolExecutor(max_workers=4)

        # 注册清理函数
        atexit.register(self.cleanup)

    def _create_experiment_dir(self) -> Path:
        """创建实验目录"""
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        dir_name = f'{self.experiment_name}_{timestamp}'
        experiment_dir = self.base_dir / dir_name

        # 创建目录结构
        for subdir in ['logs', 'checkpoints', 'results', 'artifacts', 'figures']:
            (experiment_dir / subdir).mkdir(parents=True, exist_ok=True)

        return experiment_dir

    def _save_config(self) -> None:
        """保存配置文件"""
        config_file = self.experiment_dir / 'config.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None,
                    commit: bool = True) -> None:
        """记录指标"""
        # 格式化指标字符串
        step_str = f"Step {step} - " if step is not None else ""
        metrics_str = " - ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"{step_str}{metrics_str}")

        # 添加到指标收集器
        for name, value in metrics.items():
            self.metrics_collector.add_metric(name, value)

        # 保存指标
        if commit:
            self._save_metrics()

    def _save_metrics(self) -> None:
        """保存指标到文件"""
        metrics_file = self.experiment_dir / 'metrics.json'
        with metrics_file.open('w') as f:
            json.dump(self.metrics_collector.metrics, f, indent=2)

    def log_artifact(self, name: str, artifact: Union[str, Path, bytes],
                     artifact_type: str = 'file') -> None:
        """记录制品"""
        artifact_dir = self.experiment_dir / 'artifacts'

        def _save_artifact():
            if artifact_type == 'file':
                if isinstance(artifact, (str, Path)):
                    import shutil
                    shutil.copy2(artifact, artifact_dir / name)
                else:
                    with open(artifact_dir / name, 'wb') as f:
                        f.write(artifact)

            self.logger.info(f"Saved artifact: {name}")

        # 异步保存
        self._executor.submit(_save_artifact)

    def log_figure(self, name: str, figure: 'matplotlib.figure.Figure') -> None:
        """保存matplotlib图表"""
        figure_dir = self.experiment_dir / 'figures'

        def _save_figure():
            figure.savefig(figure_dir / f'{name}.png', dpi=300, bbox_inches='tight')
            figure.close()
            self.logger.info(f"Saved figure: {name}")

        # 异步保存
        self._executor.submit(_save_figure)

    def cleanup(self) -> None:
        """清理资源"""
        self.log_manager.cleanup()
        self._executor.shutdown(wait=True)

    def __enter__(self) -> 'ExperimentLogger':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.cleanup()


def setup_logger(name: str, log_dir: Optional[PathLike] = None,
                 level: str = 'INFO', **kwargs) -> logging.Logger:
    """快速设置日志器"""
    if log_dir is None:
        log_dir = Path.cwd() / 'logs'

    config = LogConfig(
        name=name,
        log_dir=log_dir,
        level=level,
        **kwargs
    )

    log_manager = LogManager(name, config)
    return log_manager.get_logger()