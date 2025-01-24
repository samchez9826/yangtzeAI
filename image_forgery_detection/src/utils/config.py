import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging
import os
from datetime import datetime


class ConfigManager:
    """配置管理器"""

    def __init__(self, base_config_path: Union[str, Path]):
        """
        初始化配置管理器
        Args:
            base_config_path: 基础配置文件路径
        """
        self.base_config_path = Path(base_config_path)
        self.config = self._load_config(self.base_config_path)
        self.logger = logging.getLogger(__name__)

    def _load_config(self, config_path: Path) -> Dict:
        """
        加载配置文件
        Args:
            config_path: 配置文件路径
        Returns:
            配置字典
        """
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.suffix == '.yaml':
                config = yaml.safe_load(f)
            elif config_path.suffix == '.json':
                config = json.load(f)
            else:
                raise ValueError(f"不支持的配置文件格式: {config_path.suffix}")

        return config

    def update(self, override_config: Optional[Dict] = None,
               override_args: Optional[Dict] = None):
        """
        更新配置
        Args:
            override_config: 覆盖配置字典
            override_args: 命令行参数覆盖
        """
        # 更新配置字典
        if override_config:
            self._deep_update(self.config, override_config)

        # 命令行参数覆盖
        if override_args:
            for key, value in override_args.items():
                if value is not None:  # 只更新非None值
                    keys = key.split('.')
                    self._update_nested_dict(self.config, keys, value)

        # 添加运行时配置
        self._add_runtime_config()

    def _deep_update(self, base_dict: Dict, update_dict: Dict):
        """递归更新字典"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value

    def _update_nested_dict(self, d: Dict, keys: list, value: Any):
        """更新嵌套字典的值"""
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = value

    def _add_runtime_config(self):
        """添加运行时配置"""
        runtime_config = {
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'hostname': os.uname().nodename if hasattr(os, 'uname') else os.environ.get('COMPUTERNAME'),
            'pid': os.getpid()
        }

        if 'runtime' not in self.config:
            self.config['runtime'] = {}
        self.config['runtime'].update(runtime_config)

    def save(self, save_path: Union[str, Path]):
        """
        保存配置
        Args:
            save_path: 保存路径
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'w', encoding='utf-8') as f:
            if save_path.suffix == '.yaml':
                yaml.dump(self.config, f, default_flow_style=False,
                          allow_unicode=True)
            elif save_path.suffix == '.json':
                json.dump(self.config, f, indent=4, ensure_ascii=False)
            else:
                raise ValueError(f"不支持的配置文件格式: {save_path.suffix}")

        self.logger.info(f"配置已保存到: {save_path}")

    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值
        Args:
            key: 配置键，支持点号分隔的嵌套键
            default: 默认值
        Returns:
            配置值
        """
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value

    def set(self, key: str, value: Any):
        """
        设置配置值
        Args:
            key: 配置键，支持点号分隔的嵌套键
            value: 配置值
        """
        keys = key.split('.')
        self._update_nested_dict(self.config, keys, value)

    def print_config(self):
        """打印配置内容"""
        print("\n配置内容:")
        print("=" * 50)
        yaml.dump(self.config, sys.stdout, default_flow_style=False,
                  allow_unicode=True)
        print("=" * 50)


def load_config(config_path: Union[str, Path]) -> Dict:
    """
    加载配置文件的快捷函数
    Args:
        config_path: 配置文件路径
    Returns:
        配置字典
    """
    config_manager = ConfigManager(config_path)
    return config_manager.config


def merge_configs(configs: list) -> Dict:
    """
    合并多个配置
    Args:
        configs: 配置列表，后面的配置会覆盖前面的
    Returns:
        合并后的配置
    """
    merged = {}
    for config in configs:
        if isinstance(config, (str, Path)):
            config = load_config(config)
        ConfigManager._deep_update(merged, config)
    return merged


def validate_config(config: Dict, schema: Dict) -> bool:
    """
    验证配置是否符合模式
    Args:
        config: 配置字典
        schema: 模式字典
    Returns:
        是否有效
    """
    try:
        from jsonschema import validate
        validate(instance=config, schema=schema)
        return True
    except Exception as e:
        logging.error(f"配置验证失败: {str(e)}")
        return False


class ExperimentConfig(ConfigManager):
    """实验配置管理器"""

    def __init__(self, base_config_path: Union[str, Path]):
        super().__init__(base_config_path)

        # 添加实验相关配置
        self._add_experiment_config()

    def _add_experiment_config(self):
        """添加实验配置"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        experiment_config = {
            'name': self.config.get('experiment_name', 'experiment'),
            'timestamp': timestamp,
            'save_dir': str(Path('experiments') / f"{self.config.get('experiment_name', 'experiment')}_{timestamp}"),
            'seed': self.config.get('seed', 42)
        }

        if 'experiment' not in self.config:
            self.config['experiment'] = {}
        self.config['experiment'].update(experiment_config)

    def setup_experiment(self):
        """设置实验环境"""
        # 创建实验目录
        exp_dir = Path(self.config['experiment']['save_dir'])
        exp_dir.mkdir(parents=True, exist_ok=True)

        # 创建子目录
        (exp_dir / 'checkpoints').mkdir(exist_ok=True)
        (exp_dir / 'logs').mkdir(exist_ok=True)
        (exp_dir / 'results').mkdir(exist_ok=True)

        # 保存配置
        self.save(exp_dir / 'config.yaml')

        # 设置随机种子
        import torch
        import numpy as np
        import random

        seed = self.config['experiment']['seed']
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        return exp_dir