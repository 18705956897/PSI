import numpy as np
import os
from scipy.optimize import curve_fit, basinhopping


H, W = 256, 320
x, y = np.arange(W), np.arange(H)
X, Y = np.meshgrid(x, y)
CROP_SIZE = 40
S_PHOTON = 1024

NOISE_ETA1 = 25
NOISE_ETA2 = 50
NOISE_ETA3 = 100


EA_POP = 40
EA_CR = 0.9
EA_F = 0.5
EA_ITER = 10



INPUT_DIR = r".\Raw Data"
OUTPUT_DIR = r".\QE with Gaussian noise intensities"
os.makedirs(OUTPUT_DIR, exist_ok=True)



def crop_top_right(img):
    h, w = img.shape
    return img[h - CROP_SIZE:, w - CROP_SIZE:]


def add_gaussian_noise(img, eta):
    sigma = np.max(img) / eta
    noise = np.random.normal(0, sigma, img.shape)
    return img + noise


def err(gt, pred):
    mae = np.mean(np.abs(gt - pred))
    rmse = np.sqrt(np.mean((gt - pred) ** 2))
    return mae, rmse


def clip(p, bounds):
    return np.clip(p, [l for l, h in bounds], [h for l, h in bounds])


class RandomStep:
    def __init__(self, step):
        self.step = step

    def __call__(self, x):
        return x + np.random.normal(0, self.step, size=x.shape)



def model(p):
    A, B, omega, phi, fx, fy = p
    return A + B * np.cos(phi + fx * X + fy * Y)


class LightDE:
    def __init__(self, loss, bounds):
        self.loss = loss
        self.bounds = bounds
        self.pop = np.array([[np.random.uniform(l, h) for l, h in self.bounds] for _ in range(EA_POP)])
        self.fit = np.array([self.loss(x) for x in self.pop])

    def evolve(self):
        new_pop = self.pop.copy()
        n_dim = 6
        for i in range(EA_POP):
            idxs = [j for j in range(EA_POP) if j != i]
            r1, r2, r3 = np.random.choice(idxs, 3, replace=False)
            mutant = self.pop[r1] + EA_F * (self.pop[r2] - self.pop[r3])
            mutant = clip(mutant, self.bounds)
            trial = self.pop[i].copy()
            j_rand = np.random.randint(n_dim)
            for d in range(n_dim):
                if np.random.rand() < EA_CR or d == j_rand:
                    trial[d] = mutant[d]
            f_trial = self.loss(trial)
            if f_trial < self.fit[i]:
                new_pop[i] = trial
                self.fit[i] = f_trial
        self.pop = new_pop

    def best(self):
        return self.pop[np.argmin(self.fit)]


def fit_evo(S_target, x0):
    def loss(p):
        return np.sum((model(p) - S_target) ** 2)

    bounds = [(7000, 9500), (4000, 7000), (0.1, 0.3), (-np.pi, np.pi), (0, 1.6), (0, 1.6)]
    de = LightDE(loss, bounds)
    for _ in range(EA_ITER):
        de.evolve()
    res = basinhopping(loss, de.best(), niter=30,
                       minimizer_kwargs={"method": "L-BFGS-B", "bounds": bounds},
                       take_step=RandomStep(1e-3), disp=False)
    return res.x, model(res.x)


def qe_full(S, a, b, c):
    return a + b * S + c * S ** 2


class QE3D_DE:
    def __init__(self, S, I):
        self.S = S
        self.I = I
        self.bounds = [(0, 150), (0.1, 1.0), (-1e-8, 1e-8)]
        self.x0 = np.array([100.0, 0.1, 1e-6])
        noise_a = np.random.normal(0, 0.05, EA_POP)
        noise_b = np.random.normal(0, 0.00001, EA_POP)
        noise_c = np.random.normal(0, 1e-10, EA_POP)
        pop_list = []
        for i in range(EA_POP):
            a = self.x0[0] + noise_a[i]
            b = self.x0[1] + noise_b[i]
            c = self.x0[2] + noise_c[i]
            pop_list.append([a, b, c])
        self.pop = np.array(pop_list)
        self.pop = clip(self.pop, self.bounds)
        self.fit = np.array([self.loss(x) for x in self.pop])

    def loss(self, x):
        a, b, c = x
        return np.sum((a + b * self.S + c * self.S ** 2 - self.I) ** 2)

    def evolve(self):
        new_pop = self.pop.copy()
        dim = 3
        for i in range(EA_POP):
            idxs = [j for j in range(EA_POP) if j != i]
            r1, r2, r3 = np.random.choice(idxs, 3, replace=False)
            mutant = self.pop[r1] + EA_F * (self.pop[r2] - self.pop[r3])
            mutant = clip(mutant, self.bounds)
            trial = self.pop[i].copy()
            j_rand = np.random.randint(dim)
            for d in range(dim):
                if np.random.rand() < EA_CR or d == j_rand:
                    trial[d] = mutant[d]
            if self.loss(trial) < self.fit[i]:
                new_pop[i] = trial
                self.fit[i] = self.loss(trial)
        self.pop = new_pop

    def best(self):
        return self.pop[np.argmin(self.fit)]


def fit_pixel_ea(S_list, I_list):
    S = np.array(S_list)
    I = np.array(I_list)
    de = QE3D_DE(S, I)
    for _ in range(EA_ITER):
        de.evolve()
    best = de.best()

    def loss(x):
        a, b, c = x
        return np.sum((a + b * S + c * S ** 2 - I) ** 2)

    res = basinhopping(loss, best, niter=10,
                       minimizer_kwargs={"method": "L-BFGS-B", "bounds": de.bounds},
                       take_step=RandomStep([15, 1e-11, 5e-11]), disp=False)
    return tuple(res.x)


def fit_image(S1_list, I1_list, S2_list, I2_list, method):
    h, w = S1_list[0].shape
    a = np.zeros((h, w), dtype=np.float32)
    b = np.zeros((h, w), dtype=np.float32)
    c = np.zeros((h, w), dtype=np.float32)

    for y in range(h):
        for x in range(w):
            s1_pix = [s[y, x] for s in S1_list]
            i1_pix = [i[y, x] for i in I1_list]
            s2_pix = [s[y, x] for s in S2_list]
            i2_pix = [i[y, x] for i in I2_list]

            S_all = s1_pix + s2_pix
            I_all = i1_pix + i2_pix


            av, bv, cv = fit_pixel_ea(S_all, I_all)

            a[y, x] = av
            b[y, x] = bv
            c[y, x] = cv
    return a, b, c


def load_10_phases():
    S1_list, I1_list = [], []
    S2_list, I2_list = [], []

    for i in range(1, 11):
        S1 = np.load(os.path.join(INPUT_DIR, f"S1_{i}.npy"))
        I1 = np.load(os.path.join(INPUT_DIR, f"I1_{i}.npy"))
        S2 = np.load(os.path.join(INPUT_DIR, f"S2_{i}.npy"))
        I2 = np.load(os.path.join(INPUT_DIR, f"I2_{i}.npy"))

        S1_list.append(S1)
        I1_list.append(I1)
        S2_list.append(S2)
        I2_list.append(I2)

    a_true = np.load(os.path.join(INPUT_DIR, "a.npy"))
    b_true = np.load(os.path.join(INPUT_DIR, "b.npy"))
    c_true = np.load(os.path.join(INPUT_DIR, "c.npy"))

    a_true = crop_top_right(a_true)
    b_true = crop_top_right(b_true)
    c_true = crop_top_right(c_true)

    return S1_list, I1_list, S2_list, I2_list, a_true, b_true, c_true


def get_initial(I):
    data = I - np.mean(I)
    fft2 = np.fft.fft2(data)
    fft2 = np.fft.fftshift(fft2)
    amp = np.abs(fft2)
    y_max, x_max = np.unravel_index(np.argmax(amp), amp.shape)
    fx = 2 * np.pi * (x_max - W // 2) / W
    fy = 2 * np.pi * (y_max - H // 2) / H
    phi = np.angle(fft2[y_max, x_max])
    return [8192, 6000, 0.2, phi, fx, fy]


def test_with_noise(eta, S1_list, I1_list, S2_list, I2_list, a_true, b_true, c_true):
    print(f"\n{'=' * 80}")
    print(f"高斯白噪声强度 η = {eta}")
    print(f"{'=' * 80}")

    I1_noisy_list = [add_gaussian_noise(i, eta) for i in I1_list]
    I2_noisy_list = [add_gaussian_noise(i, eta) for i in I2_list]


    x0_1 = get_initial(I1_noisy_list[0])
    x0_2 = get_initial(I2_noisy_list[0])


    _, S1_ea = fit_evo(S1_list[0], x0_1)
    _, S2_ea = fit_evo(S2_list[0], x0_2)


    I1_crop_list = [crop_top_right(i) for i in I1_noisy_list]
    I2_crop_list = [crop_top_right(i) for i in I2_noisy_list]

    S1_ea, S2_ea = crop_top_right(S1_ea), crop_top_right(S2_ea)


    S1_ea_list = [S1_ea for _ in range(10)]
    S2_ea_list = [S2_ea for _ in range(10)]

    aE, bE, cE = fit_image(S1_ea_list, I1_crop_list, S2_ea_list, I2_crop_list, "ea")


    QE_true = a_true / S_PHOTON + b_true + c_true * S_PHOTON

    QE = aE / S_PHOTON + bE + cE * S_PHOTON


    np.save(os.path.join(OUTPUT_DIR, f"a_eta{eta}_ea.npy"), aE)
    np.save(os.path.join(OUTPUT_DIR, f"b_eta{eta}_ea.npy"), bE)
    np.save(os.path.join(OUTPUT_DIR, f"c_eta{eta}_ea.npy"), cE)
    # np.save(os.path.join(OUTPUT_DIR, f"QE_eta{eta}_ea.npy"), QE)


    def show(name, a_, b_, c_, q_):
        ma, ra = err(a_true, a_)
        mb, rb = err(b_true, b_)
        mc, rc = err(c_true, c_)
        mq, rq = err(QE_true, q_)
        print(f"\n{name}")
        print(f"a: MAE={ma:8.6f} | RMSE={ra:8.6f}")
        print(f"b: MAE={mb:8.6f} | RMSE={rb:8.6f}")
        print(f"c: MAE={mc:10.8f} | RMSE={rc:10.8f}")
        print(f"QE:MAE={mq:7.4f} | RMSE={rq:7.4f}")

    show("EA", aE, bE, cE, QE)



if __name__ == "__main__":

    S1_list, I1_list, S2_list, I2_list, a_true, b_true, c_true = load_10_phases()

    test_with_noise(NOISE_ETA1, S1_list, I1_list, S2_list, I2_list, a_true, b_true, c_true)
    test_with_noise(NOISE_ETA2, S1_list, I1_list, S2_list, I2_list, a_true, b_true, c_true)
    test_with_noise(NOISE_ETA3, S1_list, I1_list, S2_list, I2_list, a_true, b_true, c_true)

    print(f"\n🎉 全部完成！结果已保存到：\n{OUTPUT_DIR}")



