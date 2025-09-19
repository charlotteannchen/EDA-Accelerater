"""
Parameter Optimizer - 參數優化模組
實現貝葉斯優化、DOE 和啟發式優化算法
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import json
from scipy.optimize import minimize, differential_evolution
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import optuna

class OptimizationMethod(Enum):
    BAYESIAN = "bayesian"
    DOE = "doe"
    GENETIC = "genetic"
    NELDER_MEAD = "nelder_mead"
    GRADIENT = "gradient"

@dataclass
class Parameter:
    name: str
    min_value: float
    max_value: float
    initial_value: float
    step_size: Optional[float] = None
    log_scale: bool = False

@dataclass
class OptimizationResult:
    best_parameters: Dict[str, float]
    best_objective: float
    optimization_history: List[Dict[str, Any]]
    convergence_info: Dict[str, Any]
    success: bool

class ParameterOptimizer:
    """參數優化器"""
    
    def __init__(self, parameters: List[Parameter], objective_function: Callable):
        self.parameters = {p.name: p for p in parameters}
        self.objective_function = objective_function
        self.optimization_history = []
        
    def optimize(self, method: OptimizationMethod = OptimizationMethod.BAYESIAN,
                 n_trials: int = 100, n_jobs: int = 1) -> OptimizationResult:
        """
        執行參數優化
        
        Args:
            method: 優化方法
            n_trials: 試驗次數
            n_jobs: 並行作業數
            
        Returns:
            優化結果
        """
        if method == OptimizationMethod.BAYESIAN:
            return self._bayesian_optimization(n_trials, n_jobs)
        elif method == OptimizationMethod.DOE:
            return self._doe_optimization(n_trials)
        elif method == OptimizationMethod.GENETIC:
            return self._genetic_optimization(n_trials)
        elif method == OptimizationMethod.NELDER_MEAD:
            return self._nelder_mead_optimization()
        elif method == OptimizationMethod.GRADIENT:
            return self._gradient_optimization()
        else:
            raise ValueError(f"Unknown optimization method: {method}")
    
    def _bayesian_optimization(self, n_trials: int, n_jobs: int) -> OptimizationResult:
        """貝葉斯優化"""
        def objective(trial):
            # 生成參數組合
            params = {}
            for name, param in self.parameters.items():
                if param.log_scale:
                    params[name] = trial.suggest_float(
                        name, 
                        param.min_value, 
                        param.max_value, 
                        log=True
                    )
                else:
                    params[name] = trial.suggest_float(
                        name, 
                        param.min_value, 
                        param.max_value
                    )
            
            # 評估目標函數
            objective_value = self.objective_function(params)
            
            # 記錄歷史
            self.optimization_history.append({
                'trial': len(self.optimization_history),
                'parameters': params.copy(),
                'objective': objective_value
            })
            
            return objective_value
        
        # 創建 Optuna study
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)
        
        return OptimizationResult(
            best_parameters=study.best_params,
            best_objective=study.best_value,
            optimization_history=self.optimization_history.copy(),
            convergence_info={
                'n_trials': n_trials,
                'best_trial': study.best_trial.number,
                'pruned_trials': len(study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED]))
            },
            success=True
        )
    
    def _doe_optimization(self, n_trials: int) -> OptimizationResult:
        """實驗設計優化"""
        # 生成拉丁超立方採樣
        from scipy.stats import qmc
        
        sampler = qmc.LatinHypercube(d=len(self.parameters))
        samples = sampler.random(n=n_trials)
        
        best_objective = float('inf')
        best_parameters = {}
        
        for i, sample in enumerate(samples):
            # 將採樣轉換為參數值
            params = {}
            for j, (name, param) in enumerate(self.parameters.items()):
                if param.log_scale:
                    params[name] = np.exp(
                        np.log(param.min_value) + 
                        sample[j] * (np.log(param.max_value) - np.log(param.min_value))
                    )
                else:
                    params[name] = param.min_value + sample[j] * (param.max_value - param.min_value)
            
            # 評估目標函數
            objective_value = self.objective_function(params)
            
            # 記錄歷史
            self.optimization_history.append({
                'trial': i,
                'parameters': params.copy(),
                'objective': objective_value
            })
            
            # 更新最佳結果
            if objective_value < best_objective:
                best_objective = objective_value
                best_parameters = params.copy()
        
        return OptimizationResult(
            best_parameters=best_parameters,
            best_objective=best_objective,
            optimization_history=self.optimization_history.copy(),
            convergence_info={'n_trials': n_trials, 'method': 'DOE'},
            success=True
        )
    
    def _genetic_optimization(self, n_trials: int) -> OptimizationResult:
        """遺傳算法優化"""
        # 參數邊界
        bounds = [(p.min_value, p.max_value) for p in self.parameters.values()]
        
        def objective_array(x):
            params = {name: x[i] for i, (name, param) in enumerate(self.parameters.items())}
            return self.objective_function(params)
        
        # 執行遺傳算法
        result = differential_evolution(
            objective_array, 
            bounds, 
            maxiter=n_trials//10,  # 調整迭代次數
            popsize=15,
            seed=42
        )
        
        # 轉換結果
        best_parameters = {name: result.x[i] for i, (name, param) in enumerate(self.parameters.items())}
        
        return OptimizationResult(
            best_parameters=best_parameters,
            best_objective=result.fun,
            optimization_history=self.optimization_history.copy(),
            convergence_info={
                'n_trials': n_trials,
                'n_iterations': result.nit,
                'n_function_evaluations': result.nfev,
                'success': result.success
            },
            success=result.success
        )
    
    def _nelder_mead_optimization(self) -> OptimizationResult:
        """Nelder-Mead 單純形優化"""
        # 初始點
        x0 = [p.initial_value for p in self.parameters.values()]
        
        # 參數邊界
        bounds = [(p.min_value, p.max_value) for p in self.parameters.values()]
        
        def objective_array(x):
            params = {name: x[i] for i, (name, param) in enumerate(self.parameters.items())}
            return self.objective_function(params)
        
        # 執行優化
        result = minimize(
            objective_array, 
            x0, 
            method='Nelder-Mead',
            bounds=bounds,
            options={'maxiter': 1000}
        )
        
        # 轉換結果
        best_parameters = {name: result.x[i] for i, (name, param) in enumerate(self.parameters.items())}
        
        return OptimizationResult(
            best_parameters=best_parameters,
            best_objective=result.fun,
            optimization_history=self.optimization_history.copy(),
            convergence_info={
                'n_iterations': result.nit,
                'n_function_evaluations': result.nfev,
                'success': result.success
            },
            success=result.success
        )
    
    def _gradient_optimization(self) -> OptimizationResult:
        """梯度優化"""
        # 初始點
        x0 = [p.initial_value for p in self.parameters.values()]
        
        # 參數邊界
        bounds = [(p.min_value, p.max_value) for p in self.parameters.values()]
        
        def objective_array(x):
            params = {name: x[i] for i, (name, param) in enumerate(self.parameters.items())}
            return self.objective_function(params)
        
        # 執行優化
        result = minimize(
            objective_array, 
            x0, 
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000}
        )
        
        # 轉換結果
        best_parameters = {name: result.x[i] for i, (name, param) in enumerate(self.parameters.items())}
        
        return OptimizationResult(
            best_parameters=best_parameters,
            best_objective=result.fun,
            optimization_history=self.optimization_history.copy(),
            convergence_info={
                'n_iterations': result.nit,
                'n_function_evaluations': result.nfev,
                'success': result.success
            },
            success=result.success
        )
    
    def create_optimization_report(self, result: OptimizationResult, 
                                  output_path: str) -> None:
        """創建優化報告"""
        report = f"""# Parameter Optimization Report

## Optimization Summary
- **Method**: {result.convergence_info.get('method', 'Unknown')}
- **Best Objective Value**: {result.best_objective:.6f}
- **Success**: {'✅ Yes' if result.success else '❌ No'}
- **Total Trials**: {len(result.optimization_history)}

## Best Parameters
"""
        
        for name, value in result.best_parameters.items():
            param = self.parameters[name]
            report += f"- **{name}**: {value:.6f} (range: {param.min_value:.6f} - {param.max_value:.6f})\n"
        
        report += "\n## Optimization History\n\n"
        
        # 創建歷史數據表格
        if result.optimization_history:
            history_df = pd.DataFrame(result.optimization_history)
            
            # 添加參數列
            for name in self.parameters.keys():
                history_df[f'param_{name}'] = history_df['parameters'].apply(lambda x: x[name])
            
            # 保存為 CSV
            csv_path = output_path.replace('.md', '_history.csv')
            history_df.to_csv(csv_path, index=False)
            report += f"詳細歷史數據已保存到: {csv_path}\n\n"
        
        # 收斂信息
        report += "## Convergence Information\n\n"
        for key, value in result.convergence_info.items():
            report += f"- **{key}**: {value}\n"
        
        # 保存報告
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)

class CircuitOptimizer:
    """電路優化器 - 專門用於電路參數優化"""
    
    def __init__(self, pdk_config: Dict[str, Any]):
        self.pdk_config = pdk_config
        
    def create_circuit_parameters(self, topology_type: str) -> List[Parameter]:
        """根據拓樸類型創建參數列表"""
        if topology_type == "two_stage_ota":
            return [
                Parameter("W1", 1e-6, 50e-6, 2.4e-6, log_scale=True),  # NMOS width
                Parameter("L1", 28e-9, 200e-9, 28e-9),  # NMOS length
                Parameter("W3", 2e-6, 100e-6, 4.8e-6, log_scale=True),  # PMOS width
                Parameter("L3", 28e-9, 200e-9, 28e-9),  # PMOS length
                Parameter("Ibias", 1e-6, 100e-6, 20e-6, log_scale=True),  # Bias current
                Parameter("Rcomp", 100, 10000, 1500),  # Compensation resistor
                Parameter("Ccomp", 0.1e-12, 10e-12, 0.8e-12, log_scale=True)  # Compensation cap
            ]
        elif topology_type == "folded_cascode":
            return [
                Parameter("W1", 1e-6, 50e-6, 2.4e-6, log_scale=True),
                Parameter("L1", 28e-9, 200e-9, 28e-9),
                Parameter("W3", 2e-6, 100e-6, 4.8e-6, log_scale=True),
                Parameter("L3", 28e-9, 200e-9, 28e-9),
                Parameter("W5", 2e-6, 100e-6, 4.8e-6, log_scale=True),
                Parameter("L5", 28e-9, 200e-9, 28e-9),
                Parameter("Ibias", 1e-6, 100e-6, 30e-6, log_scale=True)
            ]
        else:
            # 通用參數
            return [
                Parameter("W1", 1e-6, 50e-6, 2.4e-6, log_scale=True),
                Parameter("L1", 28e-9, 200e-9, 28e-9),
                Parameter("W3", 2e-6, 100e-6, 4.8e-6, log_scale=True),
                Parameter("L3", 28e-9, 200e-9, 28e-9),
                Parameter("Ibias", 1e-6, 100e-6, 20e-6, log_scale=True)
            ]
    
    def create_objective_function(self, requirements: Dict[str, Any], 
                                 measurement_results: List[Dict[str, Any]]) -> Callable:
        """創建目標函數"""
        def objective(params: Dict[str, float]) -> float:
            # 這裡應該調用實際的模擬器來評估參數
            # 暫時使用簡化的目標函數
            
            # 提取目標規格
            target_gbw = requirements.get('gbw_hz', 100e6)
            target_pm = requirements.get('pm_deg', 60)
            target_power = requirements.get('power_mw', 5.0)
            target_noise = requirements.get('noise_nv_per_sqrt_hz', 10.0)
            
            # 簡化的性能估算（實際應該通過模擬獲得）
            estimated_gbw = self._estimate_gbw(params)
            estimated_pm = self._estimate_pm(params)
            estimated_power = self._estimate_power(params)
            estimated_noise = self._estimate_noise(params)
            
            # 計算目標函數（加權平方誤差）
            gbw_error = (estimated_gbw - target_gbw) / target_gbw
            pm_error = (estimated_pm - target_pm) / target_pm
            power_error = (estimated_power - target_power) / target_power
            noise_error = (estimated_noise - target_noise) / target_noise
            
            # 加權目標函數
            weights = [1.0, 1.0, 0.5, 0.8]  # GBW, PM, Power, Noise
            objective_value = (
                weights[0] * gbw_error**2 +
                weights[1] * pm_error**2 +
                weights[2] * power_error**2 +
                weights[3] * noise_error**2
            )
            
            return objective_value
        
        return objective
    
    def _estimate_gbw(self, params: Dict[str, float]) -> float:
        """估算 GBW"""
        # 簡化的 GBW 估算公式
        gm = 2 * 400e-4 * params.get('W1', 2.4e-6) / params.get('L1', 28e-9) * 0.15  # 假設 Vov=0.15V
        cload = 1e-12  # 假設負載電容
        return gm / (2 * np.pi * cload)
    
    def _estimate_pm(self, params: Dict[str, float]) -> float:
        """估算相位邊限"""
        # 簡化的 PM 估算
        return 60 + np.random.normal(0, 5)  # 添加一些隨機性
    
    def _estimate_power(self, params: Dict[str, float]) -> float:
        """估算功耗"""
        ibias = params.get('Ibias', 20e-6)
        vdd = 1.2
        return ibias * vdd * 1000  # 轉換為 mW
    
    def _estimate_noise(self, params: Dict[str, float]) -> float:
        """估算雜訊"""
        # 簡化的雜訊估算
        return 10 + np.random.normal(0, 2)  # nV/√Hz

# 使用範例
if __name__ == "__main__":
    import yaml
    
    # 載入配置
    with open("config/pdk_config.yaml", 'r') as f:
        pdk_config = yaml.safe_load(f)
    
    # 創建電路優化器
    circuit_optimizer = CircuitOptimizer(pdk_config)
    
    # 定義參數
    parameters = circuit_optimizer.create_circuit_parameters("two_stage_ota")
    
    # 定義需求
    requirements = {
        "gbw_hz": 100e6,
        "pm_deg": 60,
        "power_mw": 5.0,
        "noise_nv_per_sqrt_hz": 10.0
    }
    
    # 創建目標函數
    objective_function = circuit_optimizer.create_objective_function(requirements, [])
    
    # 創建優化器
    optimizer = ParameterOptimizer(parameters, objective_function)
    
    # 執行優化
    result = optimizer.optimize(OptimizationMethod.BAYESIAN, n_trials=50)
    
    print("優化結果:")
    print(f"最佳目標值: {result.best_objective:.6f}")
    print("最佳參數:")
    for name, value in result.best_parameters.items():
        print(f"  {name}: {value:.6e}")
    
    # 生成報告
    optimizer.create_optimization_report(result, "optimization_report.md")
