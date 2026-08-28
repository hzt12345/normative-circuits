

import os
import sys
import json
import hashlib
import random
import platform
from typing import Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field, asdict
import logging

logger = logging.getLogger(__name__)

def set_all_seeds(seed: int = 42):

    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
        logger.info(f"NumPy random seed set to {seed}")
    except ImportError:
        logger.warning("NumPy not available, skipping numpy seed")

    try:
        import torch
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            logger.info(f"PyTorch CUDA seeds set to {seed}")
        else:
            logger.info(f"PyTorch CPU seed set to {seed}")

    except ImportError:
        logger.warning("PyTorch not available, skipping torch seed")

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        logger.info(f"TensorFlow random seed set to {seed}")
    except ImportError:
        pass

    os.environ['PYTHONHASHSEED'] = str(seed)

    logger.info(f"All random seeds set to {seed}")

@dataclass
class EnvironmentInfo:
    python_version: str = field(default_factory=lambda: platform.python_version())
    platform_system: str = field(default_factory=lambda: platform.system())
    platform_release: str = field(default_factory=lambda: platform.release())
    platform_machine: str = field(default_factory=lambda: platform.machine())
    working_directory: str = field(default_factory=os.getcwd)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class PackageVersions:
    packages: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        self._collect_versions()

    def _collect_versions(self):
        packages_to_check = [
            'torch',
            'numpy',
            'scipy',
            'pandas',
            'transformers',
            'transformer_lens',
            'plotly',
            'matplotlib',
            'seaborn',
            'tqdm',
            'openai'
        ]

        for pkg in packages_to_check:
            try:
                module = __import__(pkg)
                version = getattr(module, '__version__', 'unknown')
                self.packages[pkg] = version
            except ImportError:
                self.packages[pkg] = 'not installed'

    def to_dict(self) -> Dict[str, str]:
        return self.packages

@dataclass
class GPUInfo:
    cuda_available: bool = False
    cuda_version: Optional[str] = None
    cudnn_version: Optional[str] = None
    gpu_count: int = 0
    gpu_names: list = field(default_factory=list)
    gpu_memory: list = field(default_factory=list)

    def __post_init__(self):
        self._collect_gpu_info()

    def _collect_gpu_info(self):
        try:
            import torch
            self.cuda_available = torch.cuda.is_available()

            if self.cuda_available:
                self.cuda_version = torch.version.cuda
                self.cudnn_version = str(torch.backends.cudnn.version())
                self.gpu_count = torch.cuda.device_count()

                for i in range(self.gpu_count):
                    self.gpu_names.append(torch.cuda.get_device_name(i))
                    props = torch.cuda.get_device_properties(i)
                    self.gpu_memory.append(f"{props.total_memory / 1024**3:.1f} GB")
        except ImportError:
            pass

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class ExperimentConfig:

    experiment_name: str = "path_patching_experiment"
    experiment_version: str = "1.0.0"
    random_seed: int = 42

    model_name: str = "Qwen/Qwen2.5-7B"
    model_cache_dir: str = "../../Model"

    data_path: str = "../../data/law/data.json"
    sample_size: int = 50

    suppression_strengths: list = field(default_factory=lambda: [0.3, 0.5, 0.8])
    suppression_strategy: str = "multiply"
    top_n_heads: int = 5

    alpha: float = 0.05
    confidence_level: float = 0.95

    output_dir: str = "../../output"
    save_intermediate: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def get_hash(self) -> str:
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

class ReproducibilityManager:

    def __init__(self, config: ExperimentConfig = None):
        self.config = config or ExperimentConfig()
        self.environment = EnvironmentInfo()
        self.packages = PackageVersions()
        self.gpu_info = GPUInfo()
        self._setup_reproducibility()

    def _setup_reproducibility(self):
        set_all_seeds(self.config.random_seed)
        logger.info(f"Reproducibility initialized with seed {self.config.random_seed}")

    def get_full_metadata(self) -> Dict[str, Any]:
        return {
            "experiment_config": self.config.to_dict(),
            "config_hash": self.config.get_hash(),
            "environment": self.environment.to_dict(),
            "packages": self.packages.to_dict(),
            "gpu": self.gpu_info.to_dict(),
            "metadata_timestamp": datetime.now().isoformat()
        }

    def save_metadata(self, output_dir: str = None):
        output_dir = output_dir or self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)

        metadata = self.get_full_metadata()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"experiment_metadata_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"Metadata saved to {filepath}")
        return filepath

    def validate_environment(self, reference_metadata: Dict[str, Any]) -> Dict[str, Any]:
        current = self.get_full_metadata()
        issues = []

        ref_python = reference_metadata.get("environment", {}).get("python_version")
        if ref_python and ref_python != current["environment"]["python_version"]:
            issues.append(f"Python version mismatch: {ref_python} vs {current['environment']['python_version']}")

        ref_packages = reference_metadata.get("packages", {})
        for pkg, ref_version in ref_packages.items():
            curr_version = current["packages"].get(pkg)
            if ref_version != "not installed" and curr_version != ref_version:
                issues.append(f"{pkg} version mismatch: {ref_version} vs {curr_version}")

        ref_hash = reference_metadata.get("config_hash")
        if ref_hash and ref_hash != current["config_hash"]:
            issues.append(f"Config hash mismatch: {ref_hash} vs {current['config_hash']}")

        return {
            "is_compatible": len(issues) == 0,
            "issues": issues,
            "reference_timestamp": reference_metadata.get("metadata_timestamp"),
            "current_timestamp": current["metadata_timestamp"]
        }

    def generate_requirements_txt(self, output_path: str = "requirements.txt"):
        lines = [
            "# Auto-generated requirements for reproducibility",
            f"# Generated: {datetime.now().isoformat()}",
            f"# Python version: {self.environment.python_version}",
            ""
        ]

        for pkg, version in sorted(self.packages.packages.items()):
            if version not in ["not installed", "unknown"]:
                lines.append(f"{pkg}=={version}")

        content = "\n".join(lines)

        with open(output_path, 'w') as f:
            f.write(content)

        logger.info(f"Requirements saved to {output_path}")
        return output_path

    def print_summary(self):
        print("=" * 60)
        print("Reproducibility config summary")
        print("=" * 60)

        print(f"\nExperiment: {self.config.experiment_name}")
        print(f"Config hash: {self.config.get_hash()}")
        print(f"Random seed: {self.config.random_seed}")

        print(f"\nEnvironment:")
        print(f"  Python: {self.environment.python_version}")
        print(f"  System: {self.environment.platform_system} {self.environment.platform_release}")

        print(f"\nGPU info:")
        if self.gpu_info.cuda_available:
            print(f"  CUDA: {self.gpu_info.cuda_version}")
            print(f"  cuDNN: {self.gpu_info.cudnn_version}")
            for i, (name, mem) in enumerate(zip(self.gpu_info.gpu_names, self.gpu_info.gpu_memory)):
                print(f"  GPU {i}: {name} ({mem})")
        else:
            print("  CUDA unavailable")

        print(f"\nKey package versions:")
        important_packages = ['torch', 'numpy', 'transformer_lens', 'scipy']
        for pkg in important_packages:
            version = self.packages.packages.get(pkg, 'not installed')
            print(f"  {pkg}: {version}")

        print("=" * 60)

def create_reproducible_experiment(experiment_name: str = None,
                                   random_seed: int = 42,
                                   **kwargs) -> ReproducibilityManager:
    config = ExperimentConfig(
        experiment_name=experiment_name or "path_patching_experiment",
        random_seed=random_seed,
        **kwargs
    )

    manager = ReproducibilityManager(config)
    manager.print_summary()

    return manager

def demo():
    print("=== Reproducibility Module Demo ===\n")

    manager = create_reproducible_experiment(
        experiment_name="demo_experiment",
        random_seed=42,
        sample_size=100
    )

    metadata = manager.get_full_metadata()
    print("\nMetadata preview:")
    print(f"  Config hash: {metadata['config_hash']}")
    print(f"  Timestamp: {metadata['metadata_timestamp']}")

    print("\nVerify RNG consistency:")
    import numpy as np

    set_all_seeds(42)
    random_values_1 = [random.random() for _ in range(5)]
    numpy_values_1 = np.random.random(5).tolist()

    set_all_seeds(42)
    random_values_2 = [random.random() for _ in range(5)]
    numpy_values_2 = np.random.random(5).tolist()

    print(f"  Python random (run 1): {random_values_1[:3]}")
    print(f"  Python random (run 2): {random_values_2[:3]}")
    print(f"  Values consistent: {random_values_1 == random_values_2}")

    print(f"  NumPy random (run 1): {numpy_values_1[:3]}")
    print(f"  NumPy random (run 2): {numpy_values_2[:3]}")
    print(f"  Values consistent: {numpy_values_1 == numpy_values_2}")

if __name__ == "__main__":
    demo()
