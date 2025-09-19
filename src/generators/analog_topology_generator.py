"""
Analog Topology & Code Generator - 類比拓樸與代碼生成器
根據規格生成 SPICE netlist 和 Verilog-A 行為模型
"""

import json
import re
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np

class TopologyType(Enum):
    TWO_STAGE_OTA = "two_stage_ota"
    FOLDED_CASCODE = "folded_cascode"
    TELESCOPIC = "telescopic"
    DYNAMIC_COMPARATOR = "dynamic_comparator"
    BANDGAP = "bandgap"
    LDO = "ldo"
    PLL_VCO = "pll_vco"

@dataclass
class DeviceParams:
    name: str
    device_type: str  # nmos, pmos, resistor, capacitor
    w: float  # width in meters
    l: float  # length in meters
    m: int = 1  # multiplier
    fingers: int = 1
    vov: float = 0.0  # overdrive voltage
    gm: float = 0.0  # transconductance
    gds: float = 0.0  # output conductance

@dataclass
class TopologyChoice:
    name: str
    topology_type: TopologyType
    rationale: str
    estimated_gbw: float
    estimated_pm: float
    estimated_power: float
    estimated_area: float

class AnalogTopologyGenerator:
    """類比拓樸生成器"""
    
    def __init__(self, pdk_config: Dict[str, Any]):
        self.pdk_config = pdk_config
        self.device_constraints = pdk_config['pdk']['devices']
        
        # 拓樸模板庫
        self.topology_templates = {
            TopologyType.TWO_STAGE_OTA: self._two_stage_ota_template,
            TopologyType.FOLDED_CASCODE: self._folded_cascode_template,
            TopologyType.DYNAMIC_COMPARATOR: self._dynamic_comparator_template,
            TopologyType.BANDGAP: self._bandgap_template
        }
    
    def generate_topology(self, requirements: Dict[str, Any]) -> Tuple[TopologyChoice, str, str]:
        """
        根據需求生成拓樸選擇、SPICE netlist 和 Verilog-A 模型
        
        Args:
            requirements: 電路需求規格
            
        Returns:
            Tuple[拓樸選擇, SPICE netlist, Verilog-A 模型]
        """
        # 1. 選擇合適的拓樸
        topology_choice = self._select_topology(requirements)
        
        # 2. 計算初始 sizing
        device_params = self._calculate_initial_sizing(requirements, topology_choice)
        
        # 3. 生成 SPICE netlist
        spice_netlist = self._generate_spice_netlist(topology_choice, device_params, requirements)
        
        # 4. 生成 Verilog-A 模型
        veriloga_model = self._generate_veriloga_model(topology_choice, device_params, requirements)
        
        return topology_choice, spice_netlist, veriloga_model
    
    def _select_topology(self, requirements: Dict[str, Any]) -> TopologyChoice:
        """選擇合適的拓樸"""
        gbw_req = requirements.get('gbw_hz', 100e6)
        pm_req = requirements.get('pm_deg', 60)
        load_cap = requirements.get('load_cap_f', 1e-12)
        power_budget = requirements.get('power_mw', 5)
        
        # 根據需求選擇拓樸
        if gbw_req < 50e6 and power_budget < 2:
            return TopologyChoice(
                name="Two-Stage OTA",
                topology_type=TopologyType.TWO_STAGE_OTA,
                rationale="Low power, moderate bandwidth requirement",
                estimated_gbw=gbw_req * 1.2,
                estimated_pm=pm_req,
                estimated_power=power_budget * 0.8,
                estimated_area=0.01
            )
        elif gbw_req > 100e6:
            return TopologyChoice(
                name="Folded Cascode",
                topology_type=TopologyType.FOLDED_CASCODE,
                rationale="High bandwidth requirement",
                estimated_gbw=gbw_req * 1.1,
                estimated_pm=pm_req,
                estimated_power=power_budget,
                estimated_area=0.015
            )
        else:
            return TopologyChoice(
                name="Two-Stage OTA",
                topology_type=TopologyType.TWO_STAGE_OTA,
                rationale="Balanced performance",
                estimated_gbw=gbw_req * 1.1,
                estimated_pm=pm_req,
                estimated_power=power_budget * 0.9,
                estimated_area=0.012
            )
    
    def _calculate_initial_sizing(self, requirements: Dict[str, Any], 
                                topology: TopologyChoice) -> List[DeviceParams]:
        """計算初始器件尺寸"""
        gbw_req = requirements.get('gbw_hz', 100e6)
        load_cap = requirements.get('load_cap_f', 1e-12)
        vdd = requirements.get('vdd_v', 1.2)
        
        # 基本參數
        vov = 0.15  # 150mV overdrive
        vth_n = 0.4  # NMOS threshold
        vth_p = 0.4  # PMOS threshold
        
        # 計算所需 gm
        gm_required = 2 * np.pi * gbw_req * load_cap
        
        # 根據拓樸計算器件參數
        if topology.topology_type == TopologyType.TWO_STAGE_OTA:
            return self._size_two_stage_ota(gm_required, vov, vdd, vth_n, vth_p)
        elif topology.topology_type == TopologyType.FOLDED_CASCODE:
            return self._size_folded_cascode(gm_required, vov, vdd, vth_n, vth_p)
        else:
            return self._size_generic_ota(gm_required, vov, vdd, vth_n, vth_p)
    
    def _size_two_stage_ota(self, gm_req: float, vov: float, vdd: float, 
                           vth_n: float, vth_p: float) -> List[DeviceParams]:
        """Two-Stage OTA sizing"""
        # 假設參數
        un = 400e-4  # NMOS mobility
        up = 200e-4  # PMOS mobility
        cox = 10e-3  # Oxide capacitance
        
        # 第一級
        w1 = 2 * gm_req * vov / (un * cox)
        l1 = 0.28e-6  # 最小長度
        
        # 第二級
        w2 = w1 * 2  # 更大的驅動能力
        
        # 偏置電流
        ibias = gm_req * vov / 2
        
        return [
            DeviceParams("M1", "nmos", w1, l1, vov=vov, gm=gm_req/2),
            DeviceParams("M2", "nmos", w1, l1, vov=vov, gm=gm_req/2),
            DeviceParams("M3", "pmos", w1*2, l1, vov=vov),
            DeviceParams("M4", "pmos", w1*2, l1, vov=vov),
            DeviceParams("M5", "pmos", w2, l1, vov=vov, gm=gm_req),
            DeviceParams("M6", "nmos", w2, l1, vov=vov),
            DeviceParams("Mbias", "pmos", w1, l1, vov=vov),
        ]
    
    def _size_folded_cascode(self, gm_req: float, vov: float, vdd: float,
                           vth_n: float, vth_p: float) -> List[DeviceParams]:
        """Folded Cascode sizing"""
        # 類似 Two-Stage 但結構不同
        un = 400e-4
        up = 200e-4
        cox = 10e-3
        
        w1 = 2 * gm_req * vov / (un * cox)
        l1 = 0.28e-6
        
        return [
            DeviceParams("M1", "nmos", w1, l1, vov=vov, gm=gm_req/2),
            DeviceParams("M2", "nmos", w1, l1, vov=vov, gm=gm_req/2),
            DeviceParams("M3", "pmos", w1*1.5, l1, vov=vov),
            DeviceParams("M4", "pmos", w1*1.5, l1, vov=vov),
            DeviceParams("M5", "pmos", w1*2, l1, vov=vov),
            DeviceParams("M6", "pmos", w1*2, l1, vov=vov),
            DeviceParams("Mbias", "pmos", w1, l1, vov=vov),
        ]
    
    def _size_generic_ota(self, gm_req: float, vov: float, vdd: float,
                         vth_n: float, vth_p: float) -> List[DeviceParams]:
        """通用 OTA sizing"""
        w1 = 2 * gm_req * vov / (400e-4 * 10e-3)
        l1 = 0.28e-6
        
        return [
            DeviceParams("M1", "nmos", w1, l1, vov=vov, gm=gm_req/2),
            DeviceParams("M2", "nmos", w1, l1, vov=vov, gm=gm_req/2),
            DeviceParams("M3", "pmos", w1*2, l1, vov=vov),
            DeviceParams("M4", "pmos", w1*2, l1, vov=vov),
            DeviceParams("Mbias", "pmos", w1, l1, vov=vov),
        ]
    
    def _generate_spice_netlist(self, topology: TopologyChoice, 
                               device_params: List[DeviceParams],
                               requirements: Dict[str, Any]) -> str:
        """生成 SPICE netlist"""
        if topology.topology_type == TopologyType.TWO_STAGE_OTA:
            return self._two_stage_ota_spice(device_params, requirements)
        elif topology.topology_type == TopologyType.FOLDED_CASCODE:
            return self._folded_cascode_spice(device_params, requirements)
        else:
            return self._generic_ota_spice(device_params, requirements)
    
    def _two_stage_ota_spice(self, devices: List[DeviceParams], 
                            requirements: Dict[str, Any]) -> str:
        """Two-Stage OTA SPICE netlist"""
        vdd = requirements.get('vdd_v', 1.2)
        ibias = 20e-6  # 20μA bias current
        
        netlist = f"""* Two-Stage OTA (N28, {vdd}V)
* Generated by EDA-Accelerater

* Devices
M1 out in n1 0 nmos_lvt W={devices[0].w*1e6:.2f}u L={devices[0].l*1e9:.0f}n
M2 out bias n2 0 nmos_lvt W={devices[1].w*1e6:.2f}u L={devices[1].l*1e9:.0f}n
M3 n1 vcm vdd vdd pmos_lvt W={devices[2].w*1e6:.2f}u L={devices[2].l*1e9:.0f}n
M4 n2 vcm vdd vdd pmos_lvt W={devices[3].w*1e6:.2f}u L={devices[3].l*1e9:.0f}n
M5 out bias vdd vdd pmos_lvt W={devices[4].w*1e6:.2f}u L={devices[4].l*1e9:.0f}n
M6 out bias 0 0 nmos_lvt W={devices[5].w*1e6:.2f}u L={devices[5].l*1e9:.0f}n
Mbias bias 0 vdd vdd pmos_lvt W={devices[6].w*1e6:.2f}u L={devices[6].l*1e9:.0f}n

* Compensation
Rcomp out cmiller 1.5k
Ccomp cmiller out 0.8p

* Bias
Ibias bias 0 DC {ibias*1e6:.0f}u

* Supplies
VDD vdd 0 {vdd}
VSS 0 0 0

* Input signals
Vin in 0 DC 0 AC 1
Vcm vcm 0 DC {vdd/2}

* Load
Cload out 0 {requirements.get('load_cap_f', 1e-12)*1e12:.1f}p

* Analysis
.op
.ac dec 10 1 1G
.tran 1n 1u

.end
"""
        return netlist
    
    def _folded_cascode_spice(self, devices: List[DeviceParams],
                             requirements: Dict[str, Any]) -> str:
        """Folded Cascode SPICE netlist"""
        vdd = requirements.get('vdd_v', 1.2)
        ibias = 30e-6
        
        netlist = f"""* Folded Cascode OTA (N28, {vdd}V)
* Generated by EDA-Accelerater

* Input pair
M1 n1 in n3 0 nmos_lvt W={devices[0].w*1e6:.2f}u L={devices[0].l*1e9:.0f}n
M2 n2 bias n4 0 nmos_lvt W={devices[1].w*1e6:.2f}u L={devices[1].l*1e9:.0f}n

* Cascode devices
M3 n1 vcm vdd vdd pmos_lvt W={devices[2].w*1e6:.2f}u L={devices[2].l*1e9:.0f}n
M4 n2 vcm vdd vdd pmos_lvt W={devices[3].w*1e6:.2f}u L={devices[3].l*1e9:.0f}n
M5 out n1 vdd vdd pmos_lvt W={devices[4].w*1e6:.2f}u L={devices[4].l*1e9:.0f}n
M6 out n2 vdd vdd pmos_lvt W={devices[5].w*1e6:.2f}u L={devices[5].l*1e9:.0f}n

* Bias
Mbias bias 0 vdd vdd pmos_lvt W={devices[6].w*1e6:.2f}u L={devices[6].l*1e9:.0f}n

* Current sources
M7 n3 0 0 0 nmos_lvt W={devices[0].w*1e6:.2f}u L={devices[0].l*1e9:.0f}n
M8 n4 0 0 0 nmos_lvt W={devices[1].w*1e6:.2f}u L={devices[1].l*1e9:.0f}n

* Bias current
Ibias bias 0 DC {ibias*1e6:.0f}u

* Supplies
VDD vdd 0 {vdd}
VSS 0 0 0

* Input signals
Vin in 0 DC 0 AC 1
Vcm vcm 0 DC {vdd/2}

* Load
Cload out 0 {requirements.get('load_cap_f', 1e-12)*1e12:.1f}p

* Analysis
.op
.ac dec 10 1 1G
.tran 1n 1u

.end
"""
        return netlist
    
    def _generic_ota_spice(self, devices: List[DeviceParams],
                          requirements: Dict[str, Any]) -> str:
        """通用 OTA SPICE netlist"""
        vdd = requirements.get('vdd_v', 1.2)
        ibias = 20e-6
        
        netlist = f"""* Generic OTA (N28, {vdd}V)
* Generated by EDA-Accelerater

* Differential pair
M1 n1 in n3 0 nmos_lvt W={devices[0].w*1e6:.2f}u L={devices[0].l*1e9:.0f}n
M2 n2 bias n4 0 nmos_lvt W={devices[1].w*1e6:.2f}u L={devices[1].l*1e9:.0f}n

* Load
M3 n1 vcm vdd vdd pmos_lvt W={devices[2].w*1e6:.2f}u L={devices[2].l*1e9:.0f}n
M4 n2 vcm vdd vdd pmos_lvt W={devices[3].w*1e6:.2f}u L={devices[3].l*1e9:.0f}n

* Bias
Mbias bias 0 vdd vdd pmos_lvt W={devices[4].w*1e6:.2f}u L={devices[4].l*1e9:.0f}n

* Current sources
M5 n3 0 0 0 nmos_lvt W={devices[0].w*1e6:.2f}u L={devices[0].l*1e9:.0f}n
M6 n4 0 0 0 nmos_lvt W={devices[1].w*1e6:.2f}u L={devices[1].l*1e9:.0f}n

* Bias current
Ibias bias 0 DC {ibias*1e6:.0f}u

* Supplies
VDD vdd 0 {vdd}
VSS 0 0 0

* Input signals
Vin in 0 DC 0 AC 1
Vcm vcm 0 DC {vdd/2}

* Load
Cload out 0 {requirements.get('load_cap_f', 1e-12)*1e12:.1f}p

* Analysis
.op
.ac dec 10 1 1G
.tran 1n 1u

.end
"""
        return netlist
    
    def _generate_veriloga_model(self, topology: TopologyChoice,
                                device_params: List[DeviceParams],
                                requirements: Dict[str, Any]) -> str:
        """生成 Verilog-A 行為模型"""
        gbw = topology.estimated_gbw
        pm = topology.estimated_pm
        av = 1000  # 開環增益
        sr = 50e6  # 轉換速率
        
        model = f"""`include "constants.vams"
`include "disciplines.vams"

module {topology.name.lower().replace(' ', '_')}_behav(inp, inn, out, vdd, gnd);
  input inp, inn, vdd, gnd; 
  output out;
  electrical inp, inn, out, vdd, gnd;
  
  parameter real Av = {av:.0f};        // Open loop gain
  parameter real GBW = {gbw:.0f};      // Gain bandwidth product (Hz)
  parameter real PM = {pm:.0f};        // Phase margin (degrees)
  parameter real SR = {sr:.0f};        // Slew rate (V/s)
  parameter real Rout = 50e3;          // Output resistance (Ohm)
  parameter real Vsat = 0.55;          // Output saturation voltage (V)
  parameter real Voffset = 0.0;        // Input offset voltage (V)
  
  real vout, vin_diff, vout_ideal, vout_limited;
  real tau, omega_3db;
  
  analog begin
    // Input differential voltage
    vin_diff = V(inp) - V(inn) + Voffset;
    
    // First order model with finite bandwidth
    omega_3db = 2 * `M_PI * GBW / Av;
    tau = 1.0 / omega_3db;
    
    // Ideal output voltage
    vout_ideal = Av * vin_diff;
    
    // Apply slew rate limiting
    if (abs(ddt(vout_ideal)) > SR) begin
      vout_ideal = vout_ideal + SR * $abstime * sign(ddt(vout_ideal));
    end
    
    // Apply output saturation
    vout_limited = limit(vout_ideal, -Vsat, Vsat);
    
    // First order response
    vout = vout_limited / (1 + tau * ddt(1));
    
    // Output voltage
    V(out) <+ vout;
    
    // Output resistance
    V(out) <+ Rout * I(out);
  end
endmodule
"""
        return model
    
    def _two_stage_ota_template(self) -> Dict[str, Any]:
        """Two-Stage OTA 模板"""
        return {
            "description": "Two-stage operational transconductance amplifier",
            "applications": ["Low power", "Moderate bandwidth", "General purpose"],
            "pros": ["Simple", "Low power", "Good DC gain"],
            "cons": ["Limited bandwidth", "Compensation needed"]
        }
    
    def _folded_cascode_template(self) -> Dict[str, Any]:
        """Folded Cascode 模板"""
        return {
            "description": "Folded cascode operational transconductance amplifier",
            "applications": ["High bandwidth", "High gain", "Low noise"],
            "pros": ["High bandwidth", "Good gain", "No compensation needed"],
            "cons": ["Higher power", "More complex"]
        }
    
    def _dynamic_comparator_template(self) -> Dict[str, Any]:
        """Dynamic Comparator 模板"""
        return {
            "description": "Dynamic comparator for high-speed applications",
            "applications": ["ADC", "High speed", "Low power"],
            "pros": ["Very fast", "Low power", "Small area"],
            "cons": ["Offset sensitive", "Clock dependent"]
        }
    
    def _bandgap_template(self) -> Dict[str, Any]:
        """Bandgap 模板"""
        return {
            "description": "Bandgap reference voltage generator",
            "applications": ["Reference", "Temperature stable", "Low drift"],
            "pros": ["Temperature stable", "Low drift", "Precise"],
            "cons": ["Startup issues", "Process sensitive"]
        }

# 使用範例
if __name__ == "__main__":
    import yaml
    
    # 載入 PDK 配置
    with open("config/pdk_config.yaml", 'r') as f:
        pdk_config = yaml.safe_load(f)
    
    generator = AnalogTopologyGenerator(pdk_config)
    
    # 範例需求
    requirements = {
        "gbw_hz": 100e6,
        "pm_deg": 60,
        "load_cap_f": 1e-12,
        "power_mw": 5,
        "vdd_v": 1.2,
        "noise_nv_per_sqrt_hz": 10
    }
    
    topology, spice, veriloga = generator.generate_topology(requirements)
    
    print(f"選擇拓樸: {topology.name}")
    print(f"理由: {topology.rationale}")
    print(f"預估 GBW: {topology.estimated_gbw/1e6:.1f} MHz")
    print(f"預估功耗: {topology.estimated_power:.1f} mW")
    
    # 保存文件
    with open("generated_circuit.sp", "w") as f:
        f.write(spice)
    
    with open("generated_circuit.va", "w") as f:
        f.write(veriloga)
