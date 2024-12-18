from pathlib import Path
from segm.data.base import BaseMMSeg
from segm.data import utils
from segm.config import dataset_dir
import yaml

# SMC 데이터셋에 대한 설정 파일 경로
SMC_CONFIG_PATH = Path(__file__).parent / "config" / "smc.py"
SMC_CATS_PATH = Path(__file__).parent / "config" / "smc.yml"

class SMCSegmentation(BaseMMSeg):
    def __init__(self, image_size, crop_size, split, **kwargs):
        super().__init__(
            image_size,
            crop_size,
            split,
            SMC_CONFIG_PATH,
            **kwargs,
        )
        self.names, self.colors = self.load_classes_and_palette(SMC_CATS_PATH)
        self.n_cls = len(self.names)
        self.ignore_label = 255  # 무시할 라벨 값
        self.reduce_zero_label = False

    @staticmethod
    def load_classes_and_palette(yaml_path):
        """YAML 파일에서 클래스 이름과 색상 정보를 로드합니다."""
        with open(yaml_path, 'r') as file:
            data = yaml.safe_load(file)
        classes = [item['name'] for item in data]
        palette = [item['color'] for item in data]
        return classes, palette

    def update_default_config(self, config):
            root_dir = dataset_dir()
            path = Path(root_dir)
            if self.split == "train":
                config.data.train.split_file = path / "splits/train.txt"
            elif self.split == "val":
                config.data.val.split_file = path / "splits/val.txt"
            elif self.split == "test":
                config.data.test.split_file = path / "splits/test.txt"
            config = super().update_default_config(config)
            return config


    def test_post_process(self, labels):
        """테스트 시 후처리 단계 (필요 시 사용자 정의)."""
        return labels
