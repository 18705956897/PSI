import numpy as np
import time
from scipy.optimize import basinhopping, minimize
import os
import sys

DATA_PATH = r".\pre_C1_S2\Data80_1.npy"

DIMENSION = 6
N_PARAMS = DIMENSION

CROSSOVER_RATE = 0.9
DIFFERENTIAL_WEIGHT = 0.5
POPULATION_SIZE = 100
N_ITER_PRETRAIN = 5
BH_NITER = 10
MAX_ITER = 1000
STEP_SCALE = 0.1

print(f"尝试加载数据文件: {DATA_PATH}")
try:
    MEASURED_S = np.load(DATA_PATH).astype(np.float64)
except FileNotFoundError:
    raise FileNotFoundError(f"未找到文件 {DATA_PATH}。请检查路径。")

if MEASURED_S.ndim != 2:
    raise ValueError(f"加载的数据维度为 {MEASURED_S.ndim}，不是预期的 2D 格式。")

H, W = MEASURED_S.shape
print(f"数据加载成功，尺寸: {H}x{W}")

i_coord = np.arange(1, W + 1)
j_coord = np.arange(1, H + 1)
I_grid, J_grid = np.meshgrid(i_coord, j_coord, indexing='xy')

FLAT_MEASURED_S = MEASURED_S.flatten()
N_DATA_POINTS = FLAT_MEASURED_S.size
MAX_GRAY_VAL = np.max(MEASURED_S)


MIN_BOUNDS = np.array([
    MAX_GRAY_VAL / 3, MAX_GRAY_VAL / 4, 0.0, 0.0, -2 * np.pi, -2 * np.pi
])
MAX_BOUNDS = np.array([
    MAX_GRAY_VAL, MAX_GRAY_VAL / 2, 2.0, 2.0, 2 * np.pi, 2 * np.pi
])
PARAM_BOUNDS_6D = list(zip(MIN_BOUNDS, MAX_BOUNDS))

EMPIRICAL_REF = np.array([
    8530.0,
    5918.0,
    1.024,
    0.736,
    0.288,
    -0.5099
])

TIGHT_RANGE_DELTAS = np.array([500.0, 500.0, 0.1, 0.1, 0.5, 0.5])

TIGHT_MIN_BOUNDS = EMPIRICAL_REF - TIGHT_RANGE_DELTAS
TIGHT_MAX_BOUNDS = EMPIRICAL_REF + TIGHT_RANGE_DELTAS
TIGHT_PARAM_BOUNDS = list(zip(TIGHT_MIN_BOUNDS, TIGHT_MAX_BOUNDS))


def fringe_field_6params(params, i_grid, j_grid):

    A, B, fy_val, fx_val, omega_phase, const_phase = params

    total_phi = omega_phase + const_phase
    phase = fy_val * j_grid + fx_val * i_grid + total_phi
    S = A + B * np.sin(phase)
    return S


def objective_function(params, *args):
    I_grid, J_grid, measured_S = args
    predicted_S = fringe_field_6params(params, I_grid, J_grid)
    return np.sum((predicted_S - measured_S) ** 2)


def clip_params(params, bounds):

    for i in range(N_PARAMS):
        params[i] = np.clip(params[i], bounds[i][0], bounds[i][1])
    return params


class EvolutionaryManager:


    def __init__(self, bounds, initial_bounds, args, pop_size=POPULATION_SIZE):
        self.global_bounds = bounds
        self.initial_bounds = initial_bounds
        self.args = args
        self.pop_size = pop_size
        self.population = self._initialize_population()
        self.fitness = np.array([objective_function(p, *self.args) for p in self.population])

    def _initialize_population(self):
        pop = []
        for _ in range(self.pop_size):
            individual = [np.random.uniform(b[0], b[1]) for b in self.initial_bounds]
            pop.append(individual)
        return np.array(pop)

    def perform_population_update(self, CR, F):

        new_population = np.copy(self.population)
        for i in range(self.pop_size):
            idxs = [idx for idx in range(self.pop_size) if idx != i]
            if len(idxs) < 3: continue
            r1, r2, r3 = np.random.choice(idxs, 3, replace=False)
            base, diff1, diff2 = self.population[r1], self.population[r2], self.population[r3]

            mutant_vector = base + F * (diff1 - diff2)
            mutant_vector = clip_params(mutant_vector, self.global_bounds)

            trial_vector = np.copy(self.population[i])
            j_rand = np.random.randint(N_PARAMS)

            for j in range(N_PARAMS):
                if np.random.rand() < CR or j == j_rand:
                    trial_vector[j] = mutant_vector[j]

            trial_fitness = objective_function(trial_vector, *self.args)

            if trial_fitness < self.fitness[i]:
                new_population[i] = trial_vector
                self.fitness[i] = trial_fitness

        self.population = new_population

    def get_best_individual(self):
        best_index = np.argmin(self.fitness)
        return self.population[best_index], self.fitness[best_index]


class AdaptiveStepGenerator:


    def __init__(self, bounds, CR, F, step_scale):
        self.bounds = bounds
        self.CR = CR
        self.F = F
        self.population = None
        self.step_scale = step_scale

    def __call__(self, x):
        if self.population is None or len(self.population) < 3:
            x_new = x + np.random.uniform(-0.01 * self.step_scale, 0.01 * self.step_scale, size=N_PARAMS)
        else:
            pop_size = len(self.population)
            idxs = np.random.choice(pop_size, 3, replace=False)
            base, diff1, diff2 = self.population[idxs]

            mutant_vector = base + self.F * (diff1 - diff2)

            jump_vector = (mutant_vector - base) * self.step_scale

            trial_vector = x + jump_vector

            x_new = clip_params(trial_vector, self.bounds)

        return x_new


def Hybrid_Optimizer(args, bounds, initial_bounds, initial_guess, max_iter, step_scale):

    manager = EvolutionaryManager(bounds, initial_bounds, args)
    print("   - 正在进行 EA 预训练 (紧凑范围)...")
    for i in range(N_ITER_PRETRAIN):
        manager.perform_population_update(CROSSOVER_RATE, DIFFERENTIAL_WEIGHT)

    x0, initial_J = manager.get_best_individual()
    print(f"   - 预训练最佳初始解 J (SSE): {initial_J:.4f}")

    minimizer_kwargs = {"args": args, "bounds": bounds, "method": "L-BFGS-B", "options": {'maxiter': max_iter}}

    step_generator = AdaptiveStepGenerator(bounds, CROSSOVER_RATE, DIFFERENTIAL_WEIGHT, step_scale)
    step_generator.population = manager.population  # 传递训练好的种群

    print(f"   - 正在进行优化 (迭代次数: {BH_NITER}, 步长缩放: {step_scale})...")
    res = basinhopping(objective_function, x0,
                       niter=BH_NITER,
                       T=1.0,
                       minimizer_kwargs=minimizer_kwargs,
                       take_step=step_generator,
                       disp=False)

    final_params = clip_params(res.x, bounds)
    return final_params, res.fun

if __name__ == '__main__':
    optimizer_args = (I_grid, J_grid, MEASURED_S)

    print("\n--- 启动 稳态光场 6 参数拟合 ---")
    start_time = time.time()

    final_params, final_J = Hybrid_Optimizer(
        args=optimizer_args,
        bounds=PARAM_BOUNDS_6D,
        initial_bounds=TIGHT_PARAM_BOUNDS,
        initial_guess=EMPIRICAL_REF,
        max_iter=MAX_ITER,
        step_scale=STEP_SCALE
    )

    time_elapsed = time.time() - start_time


    predicted_S = fringe_field_6params(final_params, I_grid, J_grid)
    residuals = predicted_S - MEASURED_S
    RMSE = np.sqrt(final_J / N_DATA_POINTS)
    MAE = np.mean(np.abs(residuals))
    RRMSE = RMSE/256


    print("\n--- 拟合结果 ---")
    print(f"总耗时: {time_elapsed:.2f}s")
    print("-" * 50)
    print(f"最终拟合参数 [A, B, fy, fx, phi_omega, phi_const]:")
    print(f"A (直流偏置):      {final_params[0]:.8f}")
    print(f"B (振幅):    {final_params[1]:.8f}")
    print(f"fy (Y空间频率):     {final_params[2]:.8f}")
    print(f"fx (X空间频率):     {final_params[3]:.8f}")
    print(f"phi_omega (时间频率相移): {final_params[4]:.8f}")
    print(f"phi_const (常数相移):   {final_params[5]:.8f}")
    print(f"总相移 (phi_omega+phi_const): {final_params[4] + final_params[5]:.8f}")
    print("-" * 50)
    print(f"拟合结果 RMSE (均方根误差): {RMSE:.6f}")
    print(f"拟合结果 Relative RMSE (相对均方根误差): {RRMSE:.6f}")
    print(f"拟合结果 MAE (平均绝对误差): {MAE:.6f}")
    print("-" * 50)
