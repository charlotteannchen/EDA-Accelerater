"""
Architecture & Partition Suggester - 功能分割與架構建議模組
根據規格生成功能方塊圖和 Analog/Digital 分割建議
"""

import json
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import networkx as nx
import matplotlib.pyplot as plt

class BlockType(Enum):
    ANALOG = "analog"
    DIGITAL = "digital"
    MIXED = "mixed"

@dataclass
class Block:
    name: str
    block_type: BlockType
    function: str
    interfaces: List[str]
    measurement_points: List[str]
    power_estimate_mw: float
    area_estimate_mm2: float

@dataclass
class Interface:
    name: str
    signal_type: str  # voltage, current, digital, clock
    range: Tuple[float, float]
    impedance: str
    bandwidth_hz: float

@dataclass
class Architecture:
    blocks: List[Block]
    interfaces: List[Interface]
    connections: List[Tuple[str, str]]  # (from_block, to_block)
    analog_blocks: List[str]
    digital_blocks: List[str]
    mixed_blocks: List[str]

class ArchitectureSuggester:
    """架構建議器"""
    
    def __init__(self):
        # 預定義的電路模組模板
        self.block_templates = {
            "ADC": {
                "analog_blocks": ["S_H", "Frontend_Amp", "Comparator", "CDAC", "Reference"],
                "digital_blocks": ["SAR_Logic", "Calibration", "Output_Buffer"],
                "mixed_blocks": ["Clock_Gen", "Control_Logic"]
            },
            "DAC": {
                "analog_blocks": ["Current_Source", "Switching_Matrix", "Output_Buffer"],
                "digital_blocks": ["Input_Register", "Decoder", "Calibration"],
                "mixed_blocks": ["Clock_Gen", "Control_Logic"]
            },
            "AMPLIFIER": {
                "analog_blocks": ["Input_Stage", "Gain_Stage", "Output_Stage", "Bias_Circuit"],
                "digital_blocks": ["Calibration", "Control"],
                "mixed_blocks": []
            }
        }
    
    def suggest_architecture(self, spec: Dict[str, Any]) -> Architecture:
        """
        根據規格建議架構
        
        Args:
            spec: 標準化規格字典
            
        Returns:
            Architecture: 建議的架構
        """
        block_type = spec.get('block', 'ADC')
        
        # 1. 生成基本方塊
        blocks = self._generate_blocks(spec, block_type)
        
        # 2. 定義介面
        interfaces = self._define_interfaces(spec, blocks)
        
        # 3. 建立連接關係
        connections = self._create_connections(blocks, block_type)
        
        # 4. 分類 Analog/Digital
        analog_blocks, digital_blocks, mixed_blocks = self._classify_blocks(blocks)
        
        return Architecture(
            blocks=blocks,
            interfaces=interfaces,
            connections=connections,
            analog_blocks=analog_blocks,
            digital_blocks=digital_blocks,
            mixed_blocks=mixed_blocks
        )
    
    def _generate_blocks(self, spec: Dict[str, Any], block_type: str) -> List[Block]:
        """生成功能方塊"""
        blocks = []
        template = self.block_templates.get(block_type, self.block_templates["ADC"])
        
        # 生成類比方塊
        for block_name in template["analog_blocks"]:
            block = self._create_analog_block(block_name, spec)
            blocks.append(block)
        
        # 生成數位方塊
        for block_name in template["digital_blocks"]:
            block = self._create_digital_block(block_name, spec)
            blocks.append(block)
        
        # 生成混合信號方塊
        for block_name in template["mixed_blocks"]:
            block = self._create_mixed_block(block_name, spec)
            blocks.append(block)
        
        return blocks
    
    def _create_analog_block(self, name: str, spec: Dict[str, Any]) -> Block:
        """創建類比方塊"""
        block_configs = {
            "S_H": {
                "function": "Sample and Hold",
                "interfaces": ["vin", "vout", "clk"],
                "measurement_points": ["vin", "vout", "clk"],
                "power_estimate_mw": 0.5,
                "area_estimate_mm2": 0.01
            },
            "Frontend_Amp": {
                "function": "Frontend Amplifier",
                "interfaces": ["vin", "vout", "vdd", "gnd"],
                "measurement_points": ["vin", "vout", "bias"],
                "power_estimate_mw": 2.0,
                "area_estimate_mm2": 0.05
            },
            "Comparator": {
                "function": "Comparator",
                "interfaces": ["vinp", "vinn", "vout", "clk"],
                "measurement_points": ["vinp", "vinn", "vout"],
                "power_estimate_mw": 1.0,
                "area_estimate_mm2": 0.02
            },
            "CDAC": {
                "function": "Capacitive DAC",
                "interfaces": ["d_in", "vout", "vref"],
                "measurement_points": ["vout", "vref"],
                "power_estimate_mw": 0.3,
                "area_estimate_mm2": 0.1
            },
            "Reference": {
                "function": "Reference Generator",
                "interfaces": ["vref", "vdd", "gnd"],
                "measurement_points": ["vref"],
                "power_estimate_mw": 1.0,
                "area_estimate_mm2": 0.03
            }
        }
        
        config = block_configs.get(name, {
            "function": f"Analog {name}",
            "interfaces": ["in", "out"],
            "measurement_points": ["in", "out"],
            "power_estimate_mw": 0.5,
            "area_estimate_mm2": 0.01
        })
        
        return Block(
            name=name,
            block_type=BlockType.ANALOG,
            function=config["function"],
            interfaces=config["interfaces"],
            measurement_points=config["measurement_points"],
            power_estimate_mw=config["power_estimate_mw"],
            area_estimate_mm2=config["area_estimate_mm2"]
        )
    
    def _create_digital_block(self, name: str, spec: Dict[str, Any]) -> Block:
        """創建數位方塊"""
        block_configs = {
            "SAR_Logic": {
                "function": "Successive Approximation Register",
                "interfaces": ["d_in", "d_out", "clk", "start"],
                "measurement_points": ["d_out", "clk"],
                "power_estimate_mw": 1.0,
                "area_estimate_mm2": 0.02
            },
            "Calibration": {
                "function": "Digital Calibration",
                "interfaces": ["d_in", "d_out", "clk"],
                "measurement_points": ["d_out"],
                "power_estimate_mw": 0.5,
                "area_estimate_mm2": 0.01
            },
            "Output_Buffer": {
                "function": "Output Buffer",
                "interfaces": ["d_in", "d_out", "clk"],
                "measurement_points": ["d_out"],
                "power_estimate_mw": 0.3,
                "area_estimate_mm2": 0.005
            }
        }
        
        config = block_configs.get(name, {
            "function": f"Digital {name}",
            "interfaces": ["d_in", "d_out", "clk"],
            "measurement_points": ["d_out"],
            "power_estimate_mw": 0.5,
            "area_estimate_mm2": 0.01
        })
        
        return Block(
            name=name,
            block_type=BlockType.DIGITAL,
            function=config["function"],
            interfaces=config["interfaces"],
            measurement_points=config["measurement_points"],
            power_estimate_mw=config["power_estimate_mw"],
            area_estimate_mm2=config["area_estimate_mm2"]
        )
    
    def _create_mixed_block(self, name: str, spec: Dict[str, Any]) -> Block:
        """創建混合信號方塊"""
        return Block(
            name=name,
            block_type=BlockType.MIXED,
            function=f"Mixed Signal {name}",
            interfaces=["analog_in", "digital_out", "clk"],
            measurement_points=["analog_in", "digital_out"],
            power_estimate_mw=0.8,
            area_estimate_mm2=0.015
        )
    
    def _define_interfaces(self, spec: Dict[str, Any], blocks: List[Block]) -> List[Interface]:
        """定義介面規格"""
        interfaces = []
        
        # 主要信號介面
        interfaces.append(Interface(
            name="vin",
            signal_type="voltage",
            range=(-spec.get('input_range_vpp', 1.0)/2, spec.get('input_range_vpp', 1.0)/2),
            impedance="high",
            bandwidth_hz=spec.get('fs_hz', 100000000) * 0.5
        ))
        
        interfaces.append(Interface(
            name="vout",
            signal_type="digital",
            range=(0, 1.2),
            impedance="low",
            bandwidth_hz=spec.get('fs_hz', 100000000)
        ))
        
        interfaces.append(Interface(
            name="clk",
            signal_type="clock",
            range=(0, 1.2),
            impedance="low",
            bandwidth_hz=spec.get('fs_hz', 100000000)
        ))
        
        interfaces.append(Interface(
            name="vref",
            signal_type="voltage",
            range=(0.6, 1.2),
            impedance="low",
            bandwidth_hz=1000000
        ))
        
        return interfaces
    
    def _create_connections(self, blocks: List[Block], block_type: str) -> List[Tuple[str, str]]:
        """建立方塊間的連接關係"""
        connections = []
        
        if block_type == "ADC":
            # ADC 典型連接
            connections.extend([
                ("S_H", "Frontend_Amp"),
                ("Frontend_Amp", "Comparator"),
                ("CDAC", "Comparator"),
                ("Comparator", "SAR_Logic"),
                ("SAR_Logic", "CDAC"),
                ("SAR_Logic", "Output_Buffer"),
                ("Reference", "CDAC"),
                ("Clock_Gen", "S_H"),
                ("Clock_Gen", "Comparator"),
                ("Clock_Gen", "SAR_Logic")
            ])
        
        return connections
    
    def _classify_blocks(self, blocks: List[Block]) -> Tuple[List[str], List[str], List[str]]:
        """分類方塊為 Analog/Digital/Mixed"""
        analog_blocks = [b.name for b in blocks if b.block_type == BlockType.ANALOG]
        digital_blocks = [b.name for b in blocks if b.block_type == BlockType.DIGITAL]
        mixed_blocks = [b.name for b in blocks if b.block_type == BlockType.MIXED]
        
        return analog_blocks, digital_blocks, mixed_blocks
    
    def generate_architecture_diagram(self, architecture: Architecture, output_path: str) -> None:
        """生成架構圖"""
        G = nx.DiGraph()
        
        # 添加節點
        for block in architecture.blocks:
            G.add_node(block.name, 
                      block_type=block.block_type.value,
                      function=block.function)
        
        # 添加邊
        for from_block, to_block in architecture.connections:
            if from_block in [b.name for b in architecture.blocks] and \
               to_block in [b.name for b in architecture.blocks]:
                G.add_edge(from_block, to_block)
        
        # 繪製圖
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # 根據方塊類型著色
        color_map = {
            'analog': 'lightblue',
            'digital': 'lightgreen', 
            'mixed': 'lightcoral'
        }
        
        colors = [color_map.get(G.nodes[node]['block_type'], 'lightgray') 
                 for node in G.nodes()]
        
        nx.draw(G, pos, 
                node_color=colors,
                node_size=2000,
                font_size=8,
                font_weight='bold',
                arrows=True,
                arrowsize=20,
                with_labels=True)
        
        plt.title("Circuit Architecture")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def export_architecture(self, architecture: Architecture, output_path: str) -> None:
        """匯出架構到 JSON"""
        arch_dict = {
            "blocks": [
                {
                    "name": b.name,
                    "type": b.block_type.value,
                    "function": b.function,
                    "interfaces": b.interfaces,
                    "measurement_points": b.measurement_points,
                    "power_estimate_mw": b.power_estimate_mw,
                    "area_estimate_mm2": b.area_estimate_mm2
                } for b in architecture.blocks
            ],
            "interfaces": [
                {
                    "name": i.name,
                    "signal_type": i.signal_type,
                    "range": i.range,
                    "impedance": i.impedance,
                    "bandwidth_hz": i.bandwidth_hz
                } for i in architecture.interfaces
            ],
            "connections": architecture.connections,
            "analog_blocks": architecture.analog_blocks,
            "digital_blocks": architecture.digital_blocks,
            "mixed_blocks": architecture.mixed_blocks
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(arch_dict, f, indent=2, ensure_ascii=False)

# 使用範例
if __name__ == "__main__":
    suggester = ArchitectureSuggester()
    
    # 範例規格
    spec = {
        "block": "ADC",
        "resolution_bits": 12,
        "fs_hz": 100000000,
        "vdd_v": 1.2,
        "input_range_vpp": 1.0,
        "power_budget_mw": 10
    }
    
    architecture = suggester.suggest_architecture(spec)
    
    print("建議架構:")
    print(f"類比方塊: {architecture.analog_blocks}")
    print(f"數位方塊: {architecture.digital_blocks}")
    print(f"混合方塊: {architecture.mixed_blocks}")
    
    # 匯出架構
    suggester.export_architecture(architecture, "architecture.json")
    suggester.generate_architecture_diagram(architecture, "architecture.png")
