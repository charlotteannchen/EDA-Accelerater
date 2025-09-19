"""
Spec Intake & Normalizer - 規格擷取與標準化模組
將自然語言規格轉換為標準化 JSON 格式
"""

import json
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import yaml
from pathlib import Path

class SpecType(Enum):
    ADC = "ADC"
    DAC = "DAC"
    AMPLIFIER = "AMPLIFIER"
    COMPARATOR = "COMPARATOR"
    OSCILLATOR = "OSCILLATOR"
    PLL = "PLL"
    LDO = "LDO"
    BANDGAP = "BANDGAP"

@dataclass
class SpecValidation:
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    confidence: float

class SpecNormalizer:
    """規格標準化器"""
    
    def __init__(self, config_path: str = "config/llm_config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 載入 LLM 配置
        self.llm_config = self.config['llm']
        self.prompts = self.config['prompts']
        
    def normalize_spec(self, spec_input: str) -> Tuple[Dict[str, Any], SpecValidation]:
        """
        將自然語言規格轉換為標準化 JSON
        
        Args:
            spec_input: 自然語言規格描述
            
        Returns:
            Tuple[標準化規格字典, 驗證結果]
        """
        # 1. 預處理規格文本
        cleaned_spec = self._preprocess_spec(spec_input)
        
        # 2. 使用 LLM 轉換為結構化格式
        structured_spec = self._llm_convert_to_json(cleaned_spec)
        
        # 3. 驗證和補全規格
        validation = self._validate_spec(structured_spec)
        
        # 4. 補全缺失的規格
        if validation.is_valid:
            structured_spec = self._complete_spec(structured_spec)
        
        return structured_spec, validation
    
    def _preprocess_spec(self, spec_text: str) -> str:
        """預處理規格文本"""
        # 移除多餘空白
        cleaned = re.sub(r'\s+', ' ', spec_text.strip())
        
        # 標準化單位表示
        unit_mappings = {
            'MHz': 'MHz',
            'mhz': 'MHz', 
            'GHz': 'GHz',
            'ghz': 'GHz',
            'V': 'V',
            'v': 'V',
            'mV': 'mV',
            'mv': 'mV',
            'mA': 'mA',
            'ma': 'mA',
            'μA': 'μA',
            'uA': 'μA',
            'pF': 'pF',
            'pf': 'pF',
            'nF': 'nF',
            'nf': 'nF',
            'dB': 'dB',
            'db': 'dB'
        }
        
        for old, new in unit_mappings.items():
            cleaned = cleaned.replace(old, new)
            
        return cleaned
    
    def _llm_convert_to_json(self, spec_text: str) -> Dict[str, Any]:
        """使用 LLM 將規格轉換為 JSON"""
        # 這裡應該調用實際的 LLM API
        # 暫時返回範例結構
        return {
            "block": "ADC",
            "resolution_bits": 12,
            "fs_hz": 100000000,
            "vdd_v": 1.2,
            "input_range_vpp": 1.0,
            "power_budget_mw": 10,
            "snr_db_min": 68,
            "enob_min": 11,
            "area_mm2_max": 0.8,
            "priority": ["ENOB", "Power", "Area"],
            "process": "N28",
            "package": "QFN",
            "io_spec": {"ref_clk": 100000000},
            "env": {"temp_c": [-40, 125], "vdd_tol_percent": 10}
        }
    
    def _validate_spec(self, spec: Dict[str, Any]) -> SpecValidation:
        """驗證規格完整性和合理性"""
        errors = []
        warnings = []
        
        # 必要欄位檢查
        required_fields = [
            'block', 'resolution_bits', 'fs_hz', 'vdd_v', 
            'input_range_vpp', 'power_budget_mw'
        ]
        
        for field in required_fields:
            if field not in spec:
                errors.append(f"Missing required field: {field}")
        
        # 數值範圍檢查
        if 'resolution_bits' in spec:
            if not (1 <= spec['resolution_bits'] <= 24):
                errors.append("Resolution bits must be between 1 and 24")
        
        if 'vdd_v' in spec:
            if not (0.8 <= spec['vdd_v'] <= 3.3):
                warnings.append("VDD voltage outside typical range")
        
        if 'fs_hz' in spec:
            if spec['fs_hz'] <= 0:
                errors.append("Sampling frequency must be positive")
        
        # 單位一致性檢查
        if 'input_range_vpp' in spec and 'vdd_v' in spec:
            if spec['input_range_vpp'] > spec['vdd_v']:
                warnings.append("Input range exceeds VDD")
        
        is_valid = len(errors) == 0
        confidence = 1.0 - (len(errors) * 0.2 + len(warnings) * 0.1)
        confidence = max(0.0, min(1.0, confidence))
        
        return SpecValidation(is_valid, errors, warnings, confidence)
    
    def _complete_spec(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """補全缺失的規格"""
        # 設定預設值
        defaults = {
            'process': 'N28',
            'package': 'QFN',
            'priority': ['ENOB', 'Power', 'Area'],
            'io_spec': {},
            'env': {'temp_c': [-40, 125], 'vdd_tol_percent': 10}
        }
        
        for key, value in defaults.items():
            if key not in spec:
                spec[key] = value
        
        # 根據其他規格推導缺失值
        if 'snr_db_min' not in spec and 'enob_min' in spec:
            # SNR ≈ 6.02 * ENOB + 1.76
            spec['snr_db_min'] = 6.02 * spec['enob_min'] + 1.76
        
        return spec
    
    def export_spec(self, spec: Dict[str, Any], output_path: str) -> None:
        """匯出規格到文件"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(spec, f, indent=2, ensure_ascii=False)
    
    def load_spec(self, input_path: str) -> Dict[str, Any]:
        """從文件載入規格"""
        with open(input_path, 'r', encoding='utf-8') as f:
            return json.load(f)

# 使用範例
if __name__ == "__main__":
    normalizer = SpecNormalizer()
    
    # 範例規格
    spec_text = """
    設計一個 12-bit ADC，採樣率 100MHz，電源電壓 1.2V，
    輸入範圍 1Vpp，功耗預算 10mW，最小 SNR 68dB，
    最小 ENOB 11，最大面積 0.8mm²
    """
    
    spec, validation = normalizer.normalize_spec(spec_text)
    
    print("標準化規格:")
    print(json.dumps(spec, indent=2, ensure_ascii=False))
    print(f"\n驗證結果: {validation}")
