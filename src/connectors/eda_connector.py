"""
EDA Connectors - EDA 工具連接器
支援 Spectre、HSPICE、NGSPICE 等模擬器
"""

import subprocess
import os
import tempfile
import json
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import yaml
from pathlib import Path

class SimulatorType(Enum):
    NGSPICE = "ngspice"
    SPECTRE = "spectre"
    HSPICE = "hspice"

@dataclass
class SimulationResult:
    success: bool
    log_file: str
    output_files: List[str]
    execution_time: float
    error_message: Optional[str] = None
    measurements: Dict[str, float] = None

class EDAConnector:
    """EDA 工具連接器基類"""
    
    def __init__(self, simulator_type: SimulatorType, config: Dict[str, Any]):
        self.simulator_type = simulator_type
        self.config = config
        self.temp_dir = tempfile.mkdtemp(prefix="eda_accelerater_")
        
    def run_simulation(self, netlist_file: str, 
                      output_dir: Optional[str] = None) -> SimulationResult:
        """執行模擬"""
        raise NotImplementedError
    
    def parse_results(self, log_file: str) -> Dict[str, float]:
        """解析模擬結果"""
        raise NotImplementedError
    
    def cleanup(self):
        """清理臨時文件"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

class NGSpiceConnector(EDAConnector):
    """NGSPICE 連接器"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(SimulatorType.NGSPICE, config)
        self.ngspice_path = config.get('path', 'ngspice')
        self.timeout = config.get('timeout', 3600)
    
    def run_simulation(self, netlist_file: str, 
                      output_dir: Optional[str] = None) -> SimulationResult:
        """執行 NGSPICE 模擬"""
        start_time = time.time()
        
        if output_dir is None:
            output_dir = self.temp_dir
        
        # 準備輸出文件路徑
        log_file = os.path.join(output_dir, "simulation.log")
        raw_file = os.path.join(output_dir, "simulation.raw")
        
        try:
            # 構建 NGSPICE 命令
            cmd = [
                self.ngspice_path,
                "-b",  # batch mode
                "-r", raw_file,  # raw output file
                "-o", log_file,  # log file
                netlist_file
            ]
            
            # 執行模擬
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=output_dir
            )
            
            execution_time = time.time() - start_time
            
            # 檢查執行結果
            if result.returncode == 0:
                # 解析結果
                measurements = self.parse_results(log_file)
                
                return SimulationResult(
                    success=True,
                    log_file=log_file,
                    output_files=[raw_file, log_file],
                    execution_time=execution_time,
                    measurements=measurements
                )
            else:
                return SimulationResult(
                    success=False,
                    log_file=log_file,
                    output_files=[],
                    execution_time=execution_time,
                    error_message=result.stderr
                )
                
        except subprocess.TimeoutExpired:
            return SimulationResult(
                success=False,
                log_file=log_file,
                output_files=[],
                execution_time=self.timeout,
                error_message="Simulation timeout"
            )
        except Exception as e:
            return SimulationResult(
                success=False,
                log_file=log_file,
                output_files=[],
                execution_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def parse_results(self, log_file: str) -> Dict[str, float]:
        """解析 NGSPICE 結果"""
        measurements = {}
        
        try:
            with open(log_file, 'r') as f:
                content = f.read()
            
            # 解析 .measure 結果
            import re
            
            # GBW 解析
            gbw_match = re.search(r'gbw\s*=\s*([\d.e+-]+)', content, re.IGNORECASE)
            if gbw_match:
                measurements['gbw'] = float(gbw_match.group(1))
            
            # PM 解析
            pm_match = re.search(r'pm\s*=\s*([\d.e+-]+)', content, re.IGNORECASE)
            if pm_match:
                measurements['pm'] = float(pm_match.group(1))
            
            # 轉換速率解析
            slew_match = re.search(r'slew_rate\s*=\s*([\d.e+-]+)', content, re.IGNORECASE)
            if slew_match:
                measurements['slew_rate'] = float(slew_match.group(1))
            
            # 雜訊解析
            noise_match = re.search(r'noise_total\s*=\s*([\d.e+-]+)', content, re.IGNORECASE)
            if noise_match:
                measurements['noise'] = float(noise_match.group(1))
            
            # THD 解析
            thd_match = re.search(r'thd\s*=\s*([\d.e+-]+)', content, re.IGNORECASE)
            if thd_match:
                measurements['thd'] = float(thd_match.group(1))
            
        except Exception as e:
            print(f"Warning: Failed to parse results from {log_file}: {e}")
        
        return measurements

class SpectreConnector(EDAConnector):
    """Spectre 連接器"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(SimulatorType.SPECTRE, config)
        self.spectre_path = config.get('path', 'spectre')
        self.timeout = config.get('timeout', 3600)
    
    def run_simulation(self, netlist_file: str, 
                      output_dir: Optional[str] = None) -> SimulationResult:
        """執行 Spectre 模擬"""
        start_time = time.time()
        
        if output_dir is None:
            output_dir = self.temp_dir
        
        # 準備輸出文件路徑
        log_file = os.path.join(output_dir, "simulation.log")
        psf_file = os.path.join(output_dir, "simulation.psf")
        
        try:
            # 構建 Spectre 命令
            cmd = [
                self.spectre_path,
                "+log", log_file,
                "+psf", psf_file,
                netlist_file
            ]
            
            # 執行模擬
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=output_dir
            )
            
            execution_time = time.time() - start_time
            
            # 檢查執行結果
            if result.returncode == 0:
                # 解析結果
                measurements = self.parse_results(log_file)
                
                return SimulationResult(
                    success=True,
                    log_file=log_file,
                    output_files=[psf_file, log_file],
                    execution_time=execution_time,
                    measurements=measurements
                )
            else:
                return SimulationResult(
                    success=False,
                    log_file=log_file,
                    output_files=[],
                    execution_time=execution_time,
                    error_message=result.stderr
                )
                
        except subprocess.TimeoutExpired:
            return SimulationResult(
                success=False,
                log_file=log_file,
                output_files=[],
                execution_time=self.timeout,
                error_message="Simulation timeout"
            )
        except Exception as e:
            return SimulationResult(
                success=False,
                log_file=log_file,
                output_files=[],
                execution_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def parse_results(self, log_file: str) -> Dict[str, float]:
        """解析 Spectre 結果"""
        measurements = {}
        
        try:
            with open(log_file, 'r') as f:
                content = f.read()
            
            # Spectre 特定的結果解析
            import re
            
            # 這裡需要根據實際的 Spectre 輸出格式來解析
            # 暫時使用與 NGSPICE 相同的解析邏輯
            
            # GBW 解析
            gbw_match = re.search(r'gbw\s*=\s*([\d.e+-]+)', content, re.IGNORECASE)
            if gbw_match:
                measurements['gbw'] = float(gbw_match.group(1))
            
            # PM 解析
            pm_match = re.search(r'pm\s*=\s*([\d.e+-]+)', content, re.IGNORECASE)
            if pm_match:
                measurements['pm'] = float(pm_match.group(1))
            
        except Exception as e:
            print(f"Warning: Failed to parse results from {log_file}: {e}")
        
        return measurements

class HSPICEConnector(EDAConnector):
    """HSPICE 連接器"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(SimulatorType.HSPICE, config)
        self.hspice_path = config.get('path', 'hspice')
        self.timeout = config.get('timeout', 3600)
    
    def run_simulation(self, netlist_file: str, 
                      output_dir: Optional[str] = None) -> SimulationResult:
        """執行 HSPICE 模擬"""
        start_time = time.time()
        
        if output_dir is None:
            output_dir = self.temp_dir
        
        # 準備輸出文件路徑
        log_file = os.path.join(output_dir, "simulation.log")
        tr0_file = os.path.join(output_dir, "simulation.tr0")
        
        try:
            # 構建 HSPICE 命令
            cmd = [
                self.hspice_path,
                "-i", netlist_file,
                "-o", log_file
            ]
            
            # 執行模擬
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=output_dir
            )
            
            execution_time = time.time() - start_time
            
            # 檢查執行結果
            if result.returncode == 0:
                # 解析結果
                measurements = self.parse_results(log_file)
                
                return SimulationResult(
                    success=True,
                    log_file=log_file,
                    output_files=[tr0_file, log_file],
                    execution_time=execution_time,
                    measurements=measurements
                )
            else:
                return SimulationResult(
                    success=False,
                    log_file=log_file,
                    output_files=[],
                    execution_time=execution_time,
                    error_message=result.stderr
                )
                
        except subprocess.TimeoutExpired:
            return SimulationResult(
                success=False,
                log_file=log_file,
                output_files=[],
                execution_time=self.timeout,
                error_message="Simulation timeout"
            )
        except Exception as e:
            return SimulationResult(
                success=False,
                log_file=log_file,
                output_files=[],
                execution_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def parse_results(self, log_file: str) -> Dict[str, float]:
        """解析 HSPICE 結果"""
        measurements = {}
        
        try:
            with open(log_file, 'r') as f:
                content = f.read()
            
            # HSPICE 特定的結果解析
            import re
            
            # 這裡需要根據實際的 HSPICE 輸出格式來解析
            # 暫時使用與 NGSPICE 相同的解析邏輯
            
            # GBW 解析
            gbw_match = re.search(r'gbw\s*=\s*([\d.e+-]+)', content, re.IGNORECASE)
            if gbw_match:
                measurements['gbw'] = float(gbw_match.group(1))
            
            # PM 解析
            pm_match = re.search(r'pm\s*=\s*([\d.e+-]+)', content, re.IGNORECASE)
            if pm_match:
                measurements['pm'] = float(pm_match.group(1))
            
        except Exception as e:
            print(f"Warning: Failed to parse results from {log_file}: {e}")
        
        return measurements

class EDAConnectorFactory:
    """EDA 連接器工廠"""
    
    @staticmethod
    def create_connector(simulator_type: SimulatorType, 
                        config: Dict[str, Any]) -> EDAConnector:
        """創建 EDA 連接器"""
        if simulator_type == SimulatorType.NGSPICE:
            return NGSpiceConnector(config)
        elif simulator_type == SimulatorType.SPECTRE:
            return SpectreConnector(config)
        elif simulator_type == SimulatorType.HSPICE:
            return HSPICEConnector(config)
        else:
            raise ValueError(f"Unsupported simulator type: {simulator_type}")
    
    @staticmethod
    def create_from_config(config_file: str) -> EDAConnector:
        """從配置文件創建連接器"""
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # 獲取默認模擬器
        default_engine = config['simulation']['default_engine']
        simulator_type = SimulatorType(default_engine)
        
        # 獲取模擬器配置
        eda_config = config['eda_tools'].get(default_engine, {})
        
        return EDAConnectorFactory.create_connector(simulator_type, eda_config)

# 使用範例
if __name__ == "__main__":
    # 載入配置
    with open("config/pdk_config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # 創建連接器
    connector = EDAConnectorFactory.create_from_config("config/pdk_config.yaml")
    
    # 範例 netlist
    netlist = """
* Test circuit
M1 out in n1 0 nmos_lvt W=2.4u L=28n
M2 out bias n2 0 nmos_lvt W=2.4u L=28n
M3 n1 vcm vdd vdd pmos_lvt W=4.8u L=28n
M4 n2 vcm vdd vdd pmos_lvt W=4.8u L=28n
Mbias bias 0 vdd vdd pmos_lvt W=2.4u L=28n
Ibias bias 0 DC 20uA
VDD vdd 0 1.2
VSS 0 0 0
Vin in 0 DC 0 AC 1
Vcm vcm 0 DC 0.6
Cload out 0 1p

.measure ac gbw when vdb(out)=0
.measure ac pm find vp(out) when vdb(out)=0

.ac dec 100 1 1e9
.end
"""
    
    # 保存 netlist
    netlist_file = "test_circuit.sp"
    with open(netlist_file, 'w') as f:
        f.write(netlist)
    
    # 執行模擬
    print("執行模擬...")
    result = connector.run_simulation(netlist_file)
    
    if result.success:
        print("✅ 模擬成功")
        print(f"執行時間: {result.execution_time:.2f} 秒")
        print("量測結果:")
        for name, value in result.measurements.items():
            print(f"  {name}: {value}")
    else:
        print("❌ 模擬失敗")
        print(f"錯誤信息: {result.error_message}")
    
    # 清理
    connector.cleanup()
    os.remove(netlist_file)
