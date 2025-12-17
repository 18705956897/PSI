import numpy as np
import time
from scipy.optimize import basinhopping, minimize
import os
from PIL import Image
import h5py
import sys

H_LR, W_LR = 256, 320
SCALE_FACTOR = 4
H_HR, W_HR = H_LR * SCALE_FACTOR, W_LR * SCALE_FACTOR
N_FRAMES = 16

DATA_FOLDER = r".\16_frames"
DN0P1_FILE_PATH = r".\dn0p1.npz"

IMNDT_FILENAMES = [f"Data80_{101 + i}.tif" for i in range(8)] + \
                  [f"Data137_{101 + i}.tif" for i in range(8)]


CROSSOVER_RATE = 0.9
DIFFERENTIAL_WEIGHT = 0.5
POPULATION_SIZE = 30
N_ITER_PRETRAIN = 6
BH_NITER = 10
N_PARAMS = 3

PARAM_BOUNDS_3D = [
    (-0.0000013, 0.0000009),
    (0.9000021, 1.0834),
    (-0.0000003002, 0.0000003002)
]



def objective_function(params, HR_matrix, I_vector):

    prediction = HR_matrix @ params
    error = prediction - I_vector
    return np.sum(error ** 2)


def clip_params(params, bounds):

    for i in range(N_PARAMS):
        params[i] = np.clip(params[i], bounds[i][0], bounds[i][1])
    return params


class EvolutionaryManager:


    def __init__(self, bounds, HR_matrix, I_vec, pop_size=POPULATION_SIZE):
        self.bounds = bounds
        self.HR_matrix = HR_matrix
        self.I_vec = I_vec
        self.pop_size = pop_size
        self.population = self._initialize_population()
        self.fitness = np.array([objective_function(p, HR_matrix, I_vec) for p in self.population])

    def _initialize_population(self):
        pop = []
        for _ in range(self.pop_size):

            individual = [np.random.uniform(b[0], b[1]) for b in self.bounds]
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
            mutant_vector = clip_params(mutant_vector, self.bounds)

            trial_vector = np.copy(self.population[i])
            j_rand = np.random.randint(N_PARAMS)

            for j in range(N_PARAMS):
                if np.random.rand() < CR or j == j_rand:
                    trial_vector[j] = mutant_vector[j]

            trial_fitness = objective_function(trial_vector, self.HR_matrix, self.I_vec)

            if trial_fitness < self.fitness[i]:
                new_population[i] = trial_vector
                self.fitness[i] = trial_fitness

        self.population = new_population

    def get_best_individual(self):
        best_index = np.argmin(self.fitness)
        return self.population[best_index], self.fitness[best_index]



class AdaptiveStepGenerator:

    def __init__(self, bounds, CR, F):
        self.bounds = bounds
        self.CR = CR
        self.F = F
        self.population = None

    def __call__(self, x):

        if self.population is None or len(self.population) < 3:

            x_new = x + np.random.uniform(-0.01, 0.01, size=N_PARAMS)
        else:
            pop_size = len(self.population)
            idxs = np.random.choice(pop_size, 3, replace=False)
            base, diff1, diff2 = self.population[idxs]

            mutant_vector = base + self.F * (diff1 - diff2)
            mutant_vector = clip_params(mutant_vector, self.bounds)

            trial_vector = np.copy(x)
            j_rand = np.random.randint(N_PARAMS)

            for j in range(N_PARAMS):
                if np.random.rand() < self.CR or j == j_rand:
                    trial_vector[j] = mutant_vector[j]

            x_new = trial_vector

        return clip_params(x_new, self.bounds)



def Hybrid_Optimizer(HR_matrix, I_vector, bounds):

    manager = EvolutionaryManager(bounds, HR_matrix, I_vector)
    for _ in range(N_ITER_PRETRAIN):
        manager.perform_population_update(CROSSOVER_RATE, DIFFERENTIAL_WEIGHT)

    x0, initial_fitness = manager.get_best_individual()


    minimizer_kwargs = {"args": (HR_matrix, I_vector), "bounds": bounds, "method": "L-BFGS-B"}
    step_generator = AdaptiveStepGenerator(bounds, CROSSOVER_RATE, DIFFERENTIAL_WEIGHT)
    step_generator.population = manager.population

    res = basinhopping(objective_function, x0,
                       niter=BH_NITER,
                       T=1.0,
                       minimizer_kwargs=minimizer_kwargs,
                       take_step=step_generator,
                       disp=False)

    final_params = clip_params(res.x, bounds)
    return final_params, res.fun



def load_input_data():


    print("1. 正在加载数据...")

    Imndt_raw = np.zeros((N_FRAMES, H_LR, W_LR), dtype=np.float64)
    try:
        for i, filename in enumerate(IMNDT_FILENAMES):
            path = os.path.join(DATA_FOLDER, filename)
            img = Image.open(path)
            Imndt_raw[i, :, :] = np.array(img).astype(np.float64)
        print(f"   - 成功加载 {N_FRAMES} 帧观测图像 Imndt ({H_LR}x{W_LR}x{N_FRAMES}).")
    except Exception as e:
        print(f"!!! 错误: 加载 Imndt 失败，请检查路径和文件。错误: {e}")
        sys.exit(1)


    expected_shape = (N_FRAMES, H_LR, W_LR)
    try:
        print("   - 正在尝试使用 numpy.load 加载 dn0p1.npz 文件...")
        with np.load(DN0P1_FILE_PATH) as data:
            keys = list(data.keys())
            if 'dn0p1' in keys:
                dn0p1_raw = data['dn0p1'].astype(np.float64)
            elif keys:
                dn0p1_raw = data[keys[0]].astype(np.float64)
            else:
                raise ValueError("NPZ 文件中未找到任何数据数组。")
        if dn0p1_raw.shape != expected_shape:
            print(f"!!! 警告: dn0p1 的形状 {dn0p1_raw.shape} 与预期 {expected_shape} 不符。")
        print(f"   - 最终 dn0p1 形状: {dn0p1_raw.shape}.")

    except Exception as e:
        print(f"!!! 错误: 加载 dn0p1 失败，请检查文件路径和文件内容。错误: {e}")
        sys.exit(1)


    print(f"2. 正在生成 {SCALE_FACTOR}x 超分辨光场 Snxy ({N_FRAMES}x{H_HR}x{W_HR})...")
    Snxy = np.zeros((N_FRAMES, H_HR, W_HR), dtype=np.float64)
    m_map, n_map = np.meshgrid(np.arange(1, H_HR + 1), np.arange(1, W_HR + 1), indexing='ij')

    for n1 in range(8):
        Snxy[n1, :, :] = 8530 + 6300 * np.sin(0.256 * m_map + 0.184 * n_map + (n1 + 101) * 0.288 + 0.150)
    for n1 in range(8):
        frame_index = n1 + 8
        Snxy[frame_index, :, :] = 8030 + 6750 * np.sin(0.330 * m_map + 0.230 * n_map + (n1 + 101) * 0.288 + 0.120)
    print("   - Snxy 生成完成。")

    return Imndt_raw, dn0p1_raw, Snxy



if __name__ == '__main__':


    Imndt_raw, dn0p1_raw, Snxy = load_input_data()
    total_pixels = H_LR * W_LR


    QE_map_hr = np.zeros((H_HR, W_HR), dtype=np.float64)
    A_map_hr = np.zeros((H_HR, W_HR), dtype=np.float64)
    B_map_hr = np.zeros((H_HR, W_HR), dtype=np.float64)
    C_map_hr = np.zeros((H_HR, W_HR), dtype=np.float64)
    I_vector = np.zeros(N_FRAMES)

    print(f"\n开始拟合")
    start_time = time.time()

    processed_count = 0
    PROGRESS_STEP = 100

    for lr_n in range(H_LR):
        for lr_m in range(W_LR):


            I_vector[:] = (Imndt_raw[:, lr_n, lr_m] - dn0p1_raw[:, lr_n, lr_m]) * 16.0

            hr_n_start, hr_m_start = lr_n * SCALE_FACTOR, lr_m * SCALE_FACTOR


            for k_row in range(SCALE_FACTOR):
                for k_col in range(SCALE_FACTOR):
                    hr_n = hr_n_start + k_row
                    hr_m = hr_m_start + k_col

                    S_k = Snxy[:, hr_n, hr_m]

                    HR_k = np.stack([
                        np.ones(N_FRAMES),
                        S_k,
                        S_k ** 2
                    ], axis=1)


                    R_k, final_fit = Hybrid_Optimizer(HR_k, I_vector, PARAM_BOUNDS_3D)
                    a, b, c = R_k[0], R_k[1], R_k[2]


                    A_map_hr[hr_n, hr_m] = a
                    B_map_hr[hr_n, hr_m] = b
                    C_map_hr[hr_n, hr_m] = c


                    S_mean = np.mean(S_k)

                    qe_val = (b + 2 * c * S_mean) / 16.0

                    QE_map_hr[hr_n, hr_m] = np.clip(qe_val, 0.0, 1.0)

            processed_count += 1
            if processed_count % PROGRESS_STEP == 0 or processed_count == total_pixels:
                progress_percent = (processed_count / total_pixels) * 100
                time_elapsed = time.time() - start_time
                print(f"   - 进度: {processed_count}/{total_pixels} 个 LR 像素已完成 ({progress_percent:.2f}%). "
                      f"耗时: {time_elapsed:.2f}s")

    time_elapsed = time.time() - start_time
    H, W = H_HR, W_HR
    print("-" * 50)
    print(f"拟合完成。总耗时: {time_elapsed:.2f}s")
    print(f"生成的 QE 尺寸: {H}x{W} (1024x1280)。")

    print("=========================================")


    try:
        coeffs_4x = np.stack([A_map_hr, B_map_hr, C_map_hr], axis=-1)
        np.savez('D:\QE.npz', Coefficients_4x=coeffs_4x)
        print(f"成功保存 a, b, c 结果: QE.npz")
    except Exception as e:
        print(f"保存 QE.npz 失败: {e}")