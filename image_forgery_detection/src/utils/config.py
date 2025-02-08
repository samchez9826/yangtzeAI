from __future__ import annotations

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, TypeVar, Generic, Callable, Tuple, Type
import logging
import os
import shutil
from datetime import datetime
import sys
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
import tempfile
from contextlib import contextmanager, suppress
import copy
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor
import fcntl
import weakref
import time

__all__ = [
    'RuntimeConfig',
    'ConfigError',
    'ValidationError',
    'ConfigValidator',
    'SchemaValidator',
    'ConfigIO',
    'ConfigUtils',
    'LRUCache',
    'ConfigWatcher',
    'ConfigHistory',
    'TypeValidator',
    'RangeValidator',
    'PatternValidator',
    'EnumValidator',
    'ConfigManager',
    'create_config_manager'
]

# 配置logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 类型定义
ConfigType = TypeVar('ConfigType')
T = TypeVar('T')


@dataclass
class RuntimeConfig:
    """运行时配置,用于记录配置环境信息

    Attributes:
        timestamp: 时间戳
        hostname: 主机名
        pid: 进程ID
        python_version: Python版本
        config_hash: 配置哈希值
        env: 环境变量
        working_dir: 工作目录
        user: 用户名
    """
    timestamp: str = field(default_factory=lambda: datetime.now().strftime('%Y%m%d_%H%M%S'))
    hostname: str = field(
        default_factory=lambda: os.uname().nodename if hasattr(os, 'uname') else os.environ.get('COMPUTERNAME',
                                                                                                'unknown')
    )
    pid: int = field(default_factory=os.getpid)
    python_version: str = field(default_factory=lambda: sys.version.split()[0])
    config_hash: Optional[str] = None
    env: Dict[str, str] = field(default_factory=lambda: dict(os.environ))
    working_dir: str = field(default_factory=os.getcwd)
    user: str = field(default_factory=lambda: os.getlogin())

    def __post_init__(self) -> None:
        """清理和验证初始化数据"""
        # 删除敏感信息
        sensitive_keys = {'PASSWORD', 'SECRET', 'TOKEN', 'KEY', 'CREDENTIAL', 'AUTH'}
        self.env = {
            k: '****' if any(s in k.upper() for s in sensitive_keys) else v
            for k, v in self.env.items()
        }

        # 初始验证
        self.validate()

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式,包含额外的系统信息

        Returns:
            包含完整配置信息的字典
        """
        return {
            **asdict(self),
            'platform': sys.platform,
            'python_implementation': sys.implementation.name,
            'cpu_count': os.cpu_count(),
            'memory_info': self._get_memory_info()
        }

    def _get_memory_info(self) -> Dict[str, float]:
        """获取系统内存信息

        Returns:
            包含内存使用信息的字典
        """
        try:
            import psutil
            vm = psutil.virtual_memory()
            return {
                'total': vm.total / (1024 ** 3),  # GB
                'available': vm.available / (1024 ** 3),
                'percent': vm.percent
            }
        except ImportError:
            return {}

    def validate(self) -> None:
        """验证配置的有效性"""
        if not self.timestamp:
            raise ValidationError("Timestamp is required")
        if not self.hostname:
            raise ValidationError("Hostname is required")
        if self.pid <= 0:
            raise ValidationError(f"Invalid PID: {self.pid}")

        # 验证Python版本格式
        import re
        if not re.match(r'^\d+\.\d+\.\d+', self.python_version):
            raise ValidationError(f"Invalid Python version format: {self.python_version}")


class ConfigError(Exception):
    """配置错误基类,提供详细的错误信息和上下文

    Attributes:
        message: 错误消息
        details: 错误详情
        timestamp: 错误发生时间
        traceback: 追踪信息
    """

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.timestamp = datetime.now().isoformat()
        self.traceback = sys.exc_info()[2]

    def __str__(self) -> str:
        """格式化错误信息

        Returns:
            格式化的错误信息字符串
        """
        error_msg = [f"ConfigError: {self.message}"]
        if self.details:
            error_msg.append("Details:")
            for k, v in self.details.items():
                error_msg.append(f"  {k}: {v}")
        return "\n".join(error_msg)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式

        Returns:
            包含错误信息的字典
        """
        return {
            'error_type': self.__class__.__name__,
            'message': self.message,
            'details': self.details,
            'timestamp': self.timestamp,
            'traceback': self.format_traceback()
        }

    def format_traceback(self) -> str:
        """格式化追踪信息

        Returns:
            格式化的追踪信息字符串
        """
        import traceback
        if self.traceback:
            return "".join(traceback.format_tb(self.traceback))
        return ""

    def log_error(self, logger: Optional[logging.Logger] = None) -> None:
        """记录错误信息到日志

        Args:
            logger: 可选的logger实例,默认使用模块logger
        """
        logger = logger or logging.getLogger(__name__)
        logger.error(str(self))
        if self.traceback:
            logger.error(self.format_traceback())


class ValidationError(ConfigError):
    """配置验证错误,提供字段级别的错误信息

    Attributes:
        field: 验证失败的字段
        expected: 期望的值
        actual: 实际的值
        validation_type: 验证类型
    """

    def __init__(self, message: str, field: Optional[str] = None,
                 expected: Any = None, actual: Any = None,
                 details: Optional[Dict[str, Any]] = None,
                 validation_type: Optional[str] = None):
        error_details = details or {}
        if field:
            error_details['field'] = field
        if expected is not None:
            error_details['expected'] = expected
        if actual is not None:
            error_details['actual'] = actual
        if validation_type:
            error_details['validation_type'] = validation_type

        super().__init__(message, error_details)
        self.field = field
        self.expected = expected
        self.actual = actual
        self.validation_type = validation_type

    def __str__(self) -> str:
        """格式化验证错误信息

        Returns:
            格式化的错误信息字符串
        """
        error_parts = [f"ValidationError: {self.message}"]
        if self.field:
            error_parts.append(f"Field: {self.field}")
        if self.validation_type:
            error_parts.append(f"Validation Type: {self.validation_type}")
        if self.expected is not None:
            error_parts.append(f"Expected: {self.expected}")
        if self.actual is not None:
            error_parts.append(f"Actual: {self.actual}")
        if self.details:
            error_parts.append(f"Details: {json.dumps(self.details, indent=2)}")
        return "\n".join(error_parts)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式

        Returns:
            包含验证错误信息的字典
        """
        error_dict = super().to_dict()
        error_dict.update({
            'field': self.field,
            'validation_type': self.validation_type,
            'expected': self.expected,
            'actual': self.actual
        })
        return error_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ValidationError':
        """从字典创建验证错误实例

        Args:
            data: 错误信息字典

        Returns:
            ValidationError实例
        """
        return cls(
            message=data['message'],
            field=data.get('field'),
            expected=data.get('expected'),
            actual=data.get('actual'),
            details=data.get('details'),
            validation_type=data.get('validation_type')
        )


class ConfigValidator(ABC):
    """配置验证器基类,提供验证框架和通用验证方法"""

    @abstractmethod
    def validate(self, config: Dict[str, Any]) -> None:
        """验证配置的有效性

        Args:
            config: 待验证的配置字典

        Raises:
            ValidationError: 验证失败时抛出
        """
        pass

    def _validate_field(self, field: str, value: Any,
                        validators: List[Callable[[Any], bool]],
                        error_messages: List[str]) -> None:
        """验证单个字段

        Args:
            field: 字段名
            value: 字段值
            validators: 验证函数列表
            error_messages: 对应的错误信息列表

        Raises:
            ValidationError: 验证失败时抛出
        """
        for validator, error_message in zip(validators, error_messages):
            if not validator(value):
                raise ValidationError(
                    message=error_message,
                    field=field,
                    actual=value
                )

    @staticmethod
    def type_check(value: Any, expected_type: Type) -> bool:
        """类型检查"""
        return isinstance(value, expected_type)

    @staticmethod
    def range_check(value: Union[int, float],
                    min_val: Optional[Union[int, float]] = None,
                    max_val: Optional[Union[int, float]] = None) -> bool:
        """范围检查"""
        if min_val is not None and value < min_val:
            return False
        if max_val is not None and value > max_val:
            return False
        return True

    @staticmethod
    def pattern_check(value: str, pattern: str) -> bool:
        """模式检查"""
        import re
        return bool(re.match(pattern, value))

    @staticmethod
    def enum_check(value: Any, valid_values: List[Any]) -> bool:
        """枚举检查"""
        return value in valid_values


class TypeValidator(ConfigValidator):
    """类型验证器"""

    def __init__(self, type_specs: Dict[str, Type]):
        """初始化类型验证器

        Args:
            type_specs: 类型规范字典，键为配置路径，值为期望类型
        """
        self.type_specs = type_specs

    def validate(self, config: Dict[str, Any]) -> None:
        """验证配置类型

        Args:
            config: 配置字典

        Raises:
            ValidationError: 当类型不匹配时抛出
        """
        flat_config = ConfigUtils.flatten_dict(config)
        for key, expected_type in self.type_specs.items():
            if key in flat_config:
                value = flat_config[key]
                if not self.type_check(value, expected_type):
                    raise ValidationError(
                        f"Type mismatch for key '{key}'",
                        field=key,
                        expected=expected_type.__name__,
                        actual=type(value).__name__,
                        validation_type='type'
                    )


class RangeValidator(ConfigValidator):
    """范围验证器"""

    def __init__(self, range_specs: Dict[str, Dict[str, Union[int, float]]]):
        """初始化范围验证器

        Args:
            range_specs: 范围规范字典，键为配置路径，值为包含min和max的字典
        """
        self.range_specs = range_specs

    def validate(self, config: Dict[str, Any]) -> None:
        """验证配置值范围

        Args:
            config: 配置字典

        Raises:
            ValidationError: 当值超出范围时抛出
        """
        flat_config = ConfigUtils.flatten_dict(config)
        for key, specs in self.range_specs.items():
            if key in flat_config:
                value = flat_config[key]
                min_val = specs.get('min')
                max_val = specs.get('max')
                if not self.range_check(value, min_val, max_val):
                    raise ValidationError(
                        f"Value out of range for key '{key}'",
                        field=key,
                        expected=f"between {min_val} and {max_val}",
                        actual=value,
                        validation_type='range'
                    )


class PatternValidator(ConfigValidator):
    """模式验证器"""

    def __init__(self, pattern_specs: Dict[str, str]):
        """初始化模式验证器

        Args:
            pattern_specs: 模式规范字典，键为配置路径，值为正则表达式模式
        """
        self.pattern_specs = pattern_specs

    def validate(self, config: Dict[str, Any]) -> None:
        """验证配置值模式

        Args:
            config: 配置字典

        Raises:
            ValidationError: 当值不匹配模式时抛出
        """
        flat_config = ConfigUtils.flatten_dict(config)
        for key, pattern in self.pattern_specs.items():
            if key in flat_config:
                value = flat_config[key]
                if not isinstance(value, str):
                    raise ValidationError(
                        f"Pattern validation requires string value for key '{key}'",
                        field=key,
                        validation_type='pattern'
                    )
                if not self.pattern_check(value, pattern):
                    raise ValidationError(
                        f"Pattern mismatch for key '{key}'",
                        field=key,
                        expected=pattern,
                        actual=value,
                        validation_type='pattern'
                    )


class ConfigHistory:
    """配置历史记录管理,用于跟踪和管理配置的版本历史"""

    def __init__(self, max_versions: int = 50):
        """
        初始化配置历史记录管理器

        Args:
            max_versions: 保留的最大历史版本数
        """
        self._versions: List[Dict[str, Any]] = []
        self._max_versions = max_versions
        self._lock = threading.Lock()

    def add_version(self, config: Dict[str, Any]) -> None:
        """
        添加新的配置版本

        Args:
            config: 配置字典
        """
        with self._lock:
            # 深拷贝配置以避免引用问题
            config_copy = copy.deepcopy(config)

            # 添加版本元数据
            version_info = {
                'config': config_copy,
                'timestamp': datetime.now().isoformat(),
                'version': len(self._versions) + 1,
                'hash': ConfigUtils.compute_hash(config)
            }

            self._versions.append(version_info)

            # 限制版本数量
            if len(self._versions) > self._max_versions:
                self._versions.pop(0)

    def get_version(self, version: int) -> Optional[Dict[str, Any]]:
        """
        获取指定版本的配置

        Args:
            version: 版本号

        Returns:
            配置字典,不存在时返回None
        """
        with self._lock:
            for v in self._versions:
                if v['version'] == version:
                    return copy.deepcopy(v['config'])
        return None

    def get_versions(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        获取历史版本列表

        Args:
            limit: 返回的最大版本数

        Returns:
            历史版本列表
        """
        with self._lock:
            versions = self._versions[-limit:] if limit else self._versions
            return copy.deepcopy(versions)

    def clear(self) -> None:
        """清空历史记录"""
        with self._lock:
            self._versions.clear()

    def get_version_info(self, version: int) -> Optional[Dict[str, Any]]:
        """
        获取版本详细信息

        Args:
            version: 版本号

        Returns:
            版本信息字典
        """
        with self._lock:
            for v in self._versions:
                if v['version'] == version:
                    return copy.deepcopy(v)
        return None

    def diff_versions(self, version1: int, version2: int) -> Dict[str, Any]:
        """
        比较两个版本的差异

        Args:
            version1: 第一个版本号
            version2: 第二个版本号

        Returns:
            差异字典
        """
        config1 = self.get_version(version1)
        config2 = self.get_version(version2)

        if not config1 or not config2:
            raise ValueError("Invalid version numbers")

        return ConfigUtils.diff_dicts(config1, config2)


class EnumValidator(ConfigValidator):
    """枚举值验证器,用于验证配置值是否在指定的有效值集合中"""

    def __init__(self, enum_specs: Dict[str, List[Any]]):
        """
        初始化枚举验证器

        Args:
            enum_specs: 枚举规范字典,键为配置路径,值为有效值列表
        """
        self.enum_specs = enum_specs

    def validate(self, config: Dict[str, Any]) -> None:
        """
        验证配置值是否在枚举范围内

        Args:
            config: 配置字典

        Raises:
            ValidationError: 当值不在枚举范围时抛出
        """
        flat_config = ConfigUtils.flatten_dict(config)

        for key, valid_values in self.enum_specs.items():
            if key in flat_config:
                value = flat_config[key]
                if not self.enum_check(value, valid_values):
                    raise ValidationError(
                        f"Invalid enum value for key '{key}'",
                        field=key,
                        expected=valid_values,
                        actual=value,
                        validation_type='enum'
                    )

    def validate_with_custom_message(self, config: Dict[str, Any],
                                     messages: Dict[str, str]) -> None:
        """
        使用自定义错误消息验证配置

        Args:
            config: 配置字典
            messages: 自定义错误消息字典

        Raises:
            ValidationError: 当值不在枚举范围时抛出
        """
        flat_config = ConfigUtils.flatten_dict(config)

        for key, valid_values in self.enum_specs.items():
            if key in flat_config:
                value = flat_config[key]
                if not self.enum_check(value, valid_values):
                    message = messages.get(key, f"Invalid enum value for key '{key}'")
                    raise ValidationError(
                        message,
                        field=key,
                        expected=valid_values,
                        actual=value,
                        validation_type='enum'
                    )


class SchemaValidator(ConfigValidator):
    """JSON Schema验证器"""

    def __init__(self, schema: Dict[str, Any]):
        """初始化Schema验证器

        Args:
            schema: JSON Schema定义
        """
        self.schema = schema

    def validate(self, config: Dict[str, Any]) -> None:
        """使用JSON Schema验证配置

        Args:
            config: 配置字典

        Raises:
            ValidationError: 当配置不符合schema时抛出
        """
        try:
            from jsonschema import validate
            validate(instance=config, schema=self.schema)
        except Exception as e:
            raise ValidationError(
                message=f"Schema validation failed: {str(e)}",
                validation_type='schema'
            )


class ConfigIO:
    """配置IO操作管理"""

    def __init__(self, use_locking: bool = True, max_workers: int = 4):
        """初始化配置IO管理器

        Args:
            use_locking: 是否使用文件锁
            max_workers: 线程池工作线程数
        """
        self.use_locking = use_locking
        self._lock = threading.Lock() if use_locking else None
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._closed = False

    @contextmanager
    def file_lock(self, path: Path) -> None:
        """文件锁上下文管理器

        Args:
            path: 文件路径
        """
        if not self.use_locking:
            yield
            return

        lock_path = path.with_suffix(path.suffix + '.lock')
        lock_fd = None

        try:
            lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR)
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            yield
        finally:
            if lock_fd is not None:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                os.close(lock_fd)
            with suppress(FileNotFoundError):
                lock_path.unlink()

    def load(self, path: Union[str, Path], encoding: str = 'utf-8') -> Dict[str, Any]:
        """加载配置文件

        Args:
            path: 文件路径
            encoding: 文件编码

        Returns:
            配置字典

        Raises:
            FileNotFoundError: 文件不存在时抛出
            ConfigError: 加载失败时抛出
        """
        if self._closed:
            raise RuntimeError("ConfigIO is closed")

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with self.file_lock(path):
            try:
                with open(path, 'r', encoding=encoding) as f:
                    if path.suffix in {'.yaml', '.yml'}:
                        config = yaml.safe_load(f)
                    elif path.suffix == '.json':
                        config = json.load(f)
                    else:
                        raise ConfigError(f"Unsupported config format: {path.suffix}")
                return config or {}
            except Exception as e:
                raise ConfigError(f"Failed to load config from {path}: {str(e)}")

    @contextmanager
    def safe_save(self, path: Path) -> None:
        """安全保存上下文管理器

        Args:
            path: 保存路径
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(f'.tmp{path.suffix}')
        backup_path = path.with_suffix(f'.bak{path.suffix}')

        try:
            yield tmp_path

            # 如果目标文件存在,先备份
            if path.exists():
                self._executor.submit(lambda: shutil.copy2(str(path), str(backup_path)))

            # 重命名临时文件
            if tmp_path.exists():
                tmp_path.rename(path)

        except Exception:
            # 发生错误时恢复备份
            if backup_path.exists():
                backup_path.rename(path)
            raise
        finally:
            # 清理临时文件
            for p in (tmp_path, backup_path):
                with suppress(FileNotFoundError):
                    p.unlink()

    def save(self, config: Dict[str, Any], path: Union[str, Path],
             encoding: str = 'utf-8') -> None:
        """保存配置文件

        Args:
            config: 配置字典
            path: 保存路径
            encoding: 文件编码

        Raises:
            ConfigError: 保存失败时抛出
        """
        if self._closed:
            raise RuntimeError("ConfigIO is closed")

        path = Path(path)

        with self.file_lock(path):
            with self.safe_save(path) as tmp_path:
                try:
                    with open(tmp_path, 'w', encoding=encoding) as f:
                        if path.suffix in {'.yaml', '.yml'}:
                            yaml.dump(config, f, default_flow_style=False,
                                      allow_unicode=True, sort_keys=False)
                        elif path.suffix == '.json':
                            json.dump(config, f, indent=2, ensure_ascii=False,
                                      sort_keys=False)
                        else:
                            raise ConfigError(f"Unsupported config format: {path.suffix}")
                except Exception as e:
                    raise ConfigError(f"Failed to save config to {path}: {str(e)}")

    def close(self) -> None:
        """关闭资源"""
        self._closed = True
        self._executor.shutdown(wait=True)

    def __enter__(self) -> 'ConfigIO':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

class ConfigUtils:
    """配置工具类,提供配置处理的实用函数"""

    @staticmethod
    def deep_update(base_dict: Dict[str, Any],
                   update_dict: Dict[str, Any],
                   allow_new_keys: bool = True) -> None:
        """递归更新字典"""
        for key, value in update_dict.items():
            if not allow_new_keys and key not in base_dict:
                raise KeyError(f"Key not allowed: {key}")

            if isinstance(value, dict) and isinstance(base_dict.get(key), dict):
                ConfigUtils.deep_update(base_dict[key], value, allow_new_keys)
            else:
                base_dict[key] = copy.deepcopy(value)

    @staticmethod
    def get_nested(d: Dict[str, Any], keys: Union[str, List[str]],
                   default: Any = None, separator: str = '.') -> Any:
        """获取嵌套字典的值"""
        if isinstance(keys, str):
            keys = keys.split(separator)

        result = d
        for key in keys:
            try:
                result = result[key]
            except (KeyError, TypeError):
                return default
        return result

    @staticmethod
    def set_nested(d: Dict[str, Any], keys: Union[str, List[str]],
                   value: Any, separator: str = '.') -> None:
        """设置嵌套字典的值"""
        if isinstance(keys, str):
            keys = keys.split(separator)

        current = d
        for key in keys[:-1]:
            current = current.setdefault(key, {})
        current[keys[-1]] = value

    @staticmethod
    def flatten_dict(d: Dict[str, Any], parent_key: str = '',
                     separator: str = '.') -> Dict[str, Any]:
        """将嵌套字典展平"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{separator}{k}" if parent_key else k

            if isinstance(v, dict):
                items.extend(
                    ConfigUtils.flatten_dict(v, new_key, separator).items()
                )
            else:
                items.append((new_key, v))
        return dict(items)

    @staticmethod
    def unflatten_dict(d: Dict[str, Any], separator: str = '.') -> Dict[str, Any]:
        """将展平的字典还原为嵌套形式"""
        result = {}
        for key, value in d.items():
            parts = key.split(separator)
            target = result
            for part in parts[:-1]:
                target = target.setdefault(part, {})
            target[parts[-1]] = value
        return result

    @staticmethod
    def compute_hash(config: Dict[str, Any]) -> str:
        """计算配置的哈希值"""
        config_str = json.dumps(config, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(config_str.encode()).hexdigest()

    @staticmethod
    def filter_sensitive(config: Dict[str, Any],
                        sensitive_keys: Optional[List[str]] = None,
                        mask: str = '****') -> Dict[str, Any]:
        """过滤敏感信息"""
        if sensitive_keys is None:
            sensitive_keys = ['password', 'secret', 'token', 'key', 'credential', 'api_key']

        result = {}
        flat = ConfigUtils.flatten_dict(config)

        for key, value in flat.items():
            key_lower = key.lower()
            if any(s in key_lower for s in sensitive_keys):
                result[key] = mask
            else:
                result[key] = value

        return ConfigUtils.unflatten_dict(result)

    @staticmethod
    def diff_dicts(dict1: Dict[str, Any],
                   dict2: Dict[str, Any],
                   path: str = '') -> Dict[str, Any]:
        """比较两个字典的差异"""
        diff = {}
        all_keys = set(dict1) | set(dict2)

        for key in all_keys:
            current_path = f"{path}.{key}" if path else key

            if key not in dict1:
                diff[current_path] = {'type': 'added', 'value': dict2[key]}
            elif key not in dict2:
                diff[current_path] = {'type': 'removed', 'value': dict1[key]}
            elif dict1[key] != dict2[key]:
                if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                    nested_diff = ConfigUtils.diff_dicts(dict1[key], dict2[key], current_path)
                    diff.update(nested_diff)
                else:
                    diff[current_path] = {
                        'type': 'changed',
                        'old_value': dict1[key],
                        'new_value': dict2[key]
                    }

        return diff


class ConfigManager:
    """配置管理器,提供配置的加载、验证、缓存和监控功能"""

    def __init__(self,
                 default_config: Optional[Dict[str, Any]] = None,
                 validators: Optional[List[ConfigValidator]] = None,
                 runtime_info: bool = True,
                 cache_size: int = 100):
        """初始化配置管理器"""
        self.default_config = default_config or {}
        self.validators = validators or []
        self.runtime_info = runtime_info

        # 初始化组件
        self.io = ConfigIO(use_locking=True)
        self._cache = LRUCache(cache_size)
        self._watcher = ConfigWatcher()
        self._history = ConfigHistory()
        self._lock = threading.Lock()

        # 当前配置
        self._config: Dict[str, Any] = copy.deepcopy(self.default_config)
        if runtime_info:
            self._config['runtime'] = RuntimeConfig().to_dict()

        # 注册关闭处理
        import atexit
        atexit.register(self.cleanup)

    def load(self, path: Union[str, Path], validate: bool = True) -> Dict[str, Any]:
        """加载配置文件"""
        path = Path(path)

        # 检查缓存
        cache_key = f"{path}:{path.stat().st_mtime}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # 加载配置
        config = self.io.load(path)

        # 与默认配置合并
        ConfigUtils.deep_update(self._config, config)

        # 验证配置
        if validate:
            self.validate_config(self._config)

        # 更新缓存
        self._cache[cache_key] = self._config

        # 记录历史
        self._history.add_version(self._config)

        # 启动文件监控
        self._watcher.watch(path, self._on_config_change)

        return self._config

    def validate_config(self, config: Dict[str, Any]) -> None:
        """验证配置有效性"""
        for validator in self.validators:
            validator.validate(config)

    def save(self, path: Union[str, Path],
             config: Optional[Dict[str, Any]] = None) -> None:
        """保存配置到文件"""
        config = config or self._config
        self.io.save(config, path)
        self._history.add_version(config)

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        return ConfigUtils.get_nested(self._config, key, default)

    def set(self, key: str, value: Any, validate: bool = True) -> None:
        """设置配置值"""
        with self._lock:
            ConfigUtils.set_nested(self._config, key, value)
            if validate:
                self.validate_config(self._config)

    def _on_config_change(self, path: Path) -> None:
        """配置文件变化处理"""
        try:
            self.load(path)
        except Exception as e:
            logger.error(f"Failed to reload config: {e}")

    def get_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """获取配置历史"""
        return self._history.get_versions(limit)

    def diff_with_version(self, version: int) -> Dict[str, Any]:
        """与历史版本比较差异"""
        old_config = self._history.get_version(version)
        if not old_config:
            raise ValueError(f"Version {version} not found")
        return ConfigUtils.diff_dicts(old_config, self._config)

    def cleanup(self) -> None:
        """清理资源"""
        self._watcher.stop()
        self._cache.clear()
        self.io.close()


class ConfigWatcher:
    """配置文件监控器"""

    def __init__(self, check_interval: float = 1.0):
        """初始化监控器"""
        self._watches: Dict[Path, Tuple[float, Callable[[Path], None]]] = {}
        self._running = True
        self._check_interval = check_interval
        self._thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._lock = threading.Lock()
        self._thread.start()

    def watch(self, path: Path, callback: Callable[[Path], None]) -> None:
        """添加文件监控"""
        with self._lock:
            self._watches[path] = (path.stat().st_mtime, callback)

    def unwatch(self, path: Path) -> None:
        """移除文件监控"""
        with self._lock:
            self._watches.pop(path, None)

    def _watch_loop(self) -> None:
        """监控循环"""
        while self._running:
            try:
                with self._lock:
                    for path, (mtime, callback) in list(self._watches.items()):
                        if path.exists():
                            current_mtime = path.stat().st_mtime
                            if current_mtime > mtime:
                                self._watches[path] = (current_mtime, callback)
                                try:
                                    callback(path)
                                except Exception as e:
                                    logger.error(f"Error in file watch callback: {e}")
            except Exception as e:
                logger.error(f"Error in config watcher: {e}")
            time.sleep(self._check_interval)

    def stop(self) -> None:
        """停止监控"""
        self._running = False
        if self._thread.is_alive():
            self._thread.join()


class LRUCache:
    """LRU缓存实现"""

    def __init__(self, capacity: int):
        """初始化LRU缓存"""
        self.capacity = capacity
        self.cache: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def __contains__(self, key: str) -> bool:
        return key in self.cache

    def __getitem__(self, key: str) -> Any:
        with self._lock:
            value = self.cache.pop(key)
            self.cache[key] = value  # 移到最后
            return value

    def __setitem__(self, key: str, value: Any) -> None:
        with self._lock:
            if key in self.cache:
                self.cache.pop(key)
            elif len(self.cache) >= self.capacity:
                # 删除最早的项
                self.cache.pop(next(iter(self.cache)))
            self.cache[key] = value

    def clear(self) -> None:
        """清空缓存"""
        with self._lock:
            self.cache.clear()


def create_config_manager(
    config_path: Optional[Union[str, Path]] = None,
    default_config: Optional[Dict[str, Any]] = None,
    validators: Optional[List[ConfigValidator]] = None,
    runtime_info: bool = True,
    cache_size: int = 100
) -> ConfigManager:
    """创建配置管理器的工厂函数"""
    manager = ConfigManager(
        default_config=default_config,
        validators=validators,
        runtime_info=runtime_info,
        cache_size=cache_size
    )

    if config_path is not None:
        manager.load(config_path)

    return manager