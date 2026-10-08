import numpy as np
import matplotlib.pyplot as plt

lynx = np.array([
    [209.70, 368.42], [157.63, 332.16], [118.82, 284.21], [80.95, 224.56], [43.08, 244.44],
    [20.36, 266.67], [-4.26, 293.57], [2.37, 263.16], [-20.36, 292.40], [-39.29, 299.42],
    [-21.30, 259.65], [-50.65, 267.84], [-39.29, 242.11], [-55.38, 240.94], [-100.83, 300.58],
    [-149.11, 345.03], [-172.78, 361.40], [-189.82, 300.58], [-192.66, 225.73], [-181.30, 145.03],
    [-168.05, 104.09], [-184.14, 66.67], [-186.98, 31.58], [-183.20, 3.51], [-208.76, -4.68],
    [-197.40, -29.24], [-182.25, 44.44], [-203.08, -43.27], [-172.78, -92.40], [-131.12, -126.32],
    [-101.78, -147.37], [-74.32, -163.74], [-110.30, -224.56], [-143.43, -287.72], [-161.42, -240.94],
    [-282.60, -221.05], [-388.64, -205.85], [-370.65, -301.75], [-339.41, -397.66], [18.46, -397.66],
    [345.09, -400.00], [359.29, -378.95], [367.81, -342.69], [346.98, -362.57], [363.08, -302.92],
    [357.40, -243.27], [348.88, -266.67], [336.57, -201.17], [290.18, -135.67], [240.00, -118.13],
    [258.93, -164.91], [257.99, -228.07], [252.31, -271.35], [256.09, -333.33], [247.57, -359.06],
    [230.53, -307.60], [194.56, -238.60], [160.47, -181.29], [120.71, -149.71], [165.21, -132.16],
    [201.18, -100.58], [183.20, -99.42], [221.07, -73.68], [253.25, -24.56], [222.01, -23.39],
    [251.36, -1.17], [262.72, 24.56], [234.32, 25.73], [214.44, 42.11], [202.13, 60.82],
    [220.12, 101.75], [234.32, 160.23], [240.00, 230.41], [232.43, 316.96]
]).T


# Візуалізація
def plot_2d(X_orig, X_trans, title):
    plt.figure(figsize=(6, 6))
    plt.plot(X_orig[0, :], X_orig[1, :], label="Original", color="gray", linestyle="--")
    plt.plot(X_trans[0, :], X_trans[1, :], label="Transformed", color="blue")
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(True)
    plt.legend()
    plt.title(title)
    plt.axis('equal')
    plt.show()


# Функції перетворень
def stretch(X, a, b):
    X_copy = X.copy()
    A = np.array([[a, 0], [0, b]])
    return A @ X_copy, A

def shear(X, a, b):
    X_copy = X.copy()
    A = np.array([[1, a], [b, 1]])
    return A @ X_copy, A

def reflection(X, a, b):
    X_copy = X.copy()
    denom = a**2 + b**2
    A = (1.0 / denom) * np.array([
        [a**2 - b**2, 2*a*b],
        [2*a*b, b**2 - a**2]
    ])
    return A @ X_copy, A

def rotation(X, theta):
    X_copy = X.copy()
    rad = np.radians(theta)
    A = np.array([
        [np.cos(rad), -np.sin(rad)],
        [np.sin(rad),  np.cos(rad)]
    ])
    return A @ X_copy, A


print("--- Stretch ---")
X_str, A_str = stretch(lynx, 1.5, 0.7)
print("Matrix:\n", A_str)
plot_2d(lynx, X_str, "Stretch (1.5, 0.7)")

print("\n--- Shear ---")
X_sh, A_sh = shear(lynx, 0.5, 0)
print("Matrix:\n", A_sh)
plot_2d(lynx, X_sh, "Shear (0.5, 0)")

print("\n--- Reflection ---")
X_ref, A_ref = reflection(lynx, 1, 1)
print("Matrix:\n", A_ref)
plot_2d(lynx, X_ref, "Reflection (line y = x)")

print("\n--- Rotation ---")
X_rot, A_rot = rotation(lynx, 45)
print("Matrix:\n", A_rot)
plot_2d(lynx, X_rot, "Rotation (45 deg)")

print("\n--- Перевірка некомутативності ---")

# Матриці повороту (R) та зсуву (S)
_, R = rotation(lynx, 45)
_, S = shear(lynx, 0.5, 0)

# Порядок 1: Спочатку поворот, потім зсув -> S @ R
A1 = S @ R
X1 = A1 @ lynx

# Порядок 2: Спочатку зсув, потім поворот -> R @ S
A2 = R @ S
X2 = A2 @ lynx

print("Матриця S @ R (Поворот -> Зсув):\n", A1)
print("\nМатриця R @ S (Зсув -> Поворот):\n", A2)

print(f"\nЧи однаковий результат? {np.allclose(A1, A2)}")

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(lynx[0, :], lynx[1, :], label="Original", color="gray", linestyle="--")
plt.plot(X1[0, :], X1[1, :], label="S @ R", color="blue")
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(True)
plt.legend()
plt.title("Спочатку поворот, потім зсув (S @ R)")
plt.axis('equal')

plt.subplot(1, 2, 2)
plt.plot(lynx[0, :], lynx[1, :], label="Original", color="gray", linestyle="--")
plt.plot(X2[0, :], X2[1, :], label="R @ S", color="red")
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(True)
plt.legend()
plt.title("Спочатку зсув, потім поворот (R @ S)")
plt.axis('equal')

plt.tight_layout()
plt.show()