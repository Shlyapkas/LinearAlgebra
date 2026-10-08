import os
import kagglehub
import numpy as np
import matplotlib.pyplot as plt\


# 1. Завантаження датасету ModelNet40 через kagglehub
print("Перевірка датасету ModelNet40...")
dataset_path = kagglehub.dataset_download("balraj98/modelnet40-princeton-3d-object-dataset")

# Пошук першого .off файлу в датасеті
off_file_path = None
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith('.off'):
            off_file_path = os.path.join(root, file)
            break
    if off_file_path:
        break

print(f"Використовуємо файл: {off_file_path}\n")


# 2. Функція зчитування .off файлу
def read_off(filepath):
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    if lines[0] == 'OFF':
        header = lines[1].split()
        start_idx = 2
    else:
        header = lines[0][3:].strip().split()
        start_idx = 1

    n_verts = int(header[0])
    vertices = []
    for i in range(start_idx, start_idx + n_verts):
        vertices.append([float(x) for x in lines[i].split()])

    return np.array(vertices).T  # Розмірність (3, N)


# 3. 3D Матричні перетворення
def stretch_3d(X, a, b, c):
    A = np.array([
        [a, 0, 0],
        [0, b, 0],
        [0, 0, c]
    ])
    return A @ X, A


def rotate_x_3d(X, theta):
    rad = np.radians(theta)
    A = np.array([
        [1, 0, 0],
        [0, np.cos(rad), -np.sin(rad)],
        [0, np.sin(rad), np.cos(rad)]
    ])
    return A @ X, A


def rotate_y_3d(X, theta):
    rad = np.radians(theta)
    A = np.array([
        [np.cos(rad), 0, np.sin(rad)],
        [0, 1, 0],
        [-np.sin(rad), 0, np.cos(rad)]
    ])
    return A @ X, A


def rotate_z_3d(X, theta):
    rad = np.radians(theta)
    A = np.array([
        [np.cos(rad), -np.sin(rad), 0],
        [np.sin(rad), np.cos(rad), 0],
        [0, 0, 1]
    ])
    return A @ X, A


def shear_z_3d(X, sh_x, sh_y):
    A = np.array([
        [1, 0, sh_x],
        [0, 1, sh_y],
        [0, 0, 1]
    ])
    return A @ X, A


def reflection_xy_3d(X):
    A = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, -1]
    ])
    return A @ X, A


# 4. Візуалізація
def plot_3d(X_orig, X_trans, title="3D Transformation"):
    fig = plt.figure(figsize=(12, 6))

    all_points = np.hstack([X_orig, X_trans])
    max_range = np.array([
        all_points[0, :].max() - all_points[0, :].min(),
        all_points[1, :].max() - all_points[1, :].min(),
        all_points[2, :].max() - all_points[2, :].min()
    ]).max() / 2.0

    mid_x = (all_points[0, :].max() + all_points[0, :].min()) * 0.5
    mid_y = (all_points[1, :].max() + all_points[1, :].min()) * 0.5
    mid_z = (all_points[2, :].max() + all_points[2, :].min()) * 0.5

    # 1. Початкова модель
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(X_orig[0, :], X_orig[1, :], X_orig[2, :], color='gray', s=2, alpha=0.5)
    ax1.set_title("Original Model")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.set_xlim(mid_x - max_range, mid_x + max_range)
    ax1.set_ylim(mid_y - max_range, mid_y + max_range)
    ax1.set_zlim(mid_z - max_range, mid_z + max_range)

    # 2. Трансформована модель
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(X_trans[0, :], X_trans[1, :], X_trans[2, :], color='blue', s=2, alpha=0.5)
    ax2.set_title(title)
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    ax2.set_xlim(mid_x - max_range, mid_x + max_range)
    ax2.set_ylim(mid_y - max_range, mid_y + max_range)
    ax2.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()
    plt.show()



model_3d = read_off(off_file_path)

# 1. Розтяг (Stretch)
print("--- 1. Stretch ---")
X_str, A_str = stretch_3d(model_3d, 1.5, 0.5, 2.0)
print("Матриця:\n", A_str)
plot_3d(model_3d, X_str, "Stretch (1.5, 0.5, 2.0)")

# 2. Поворот навколо осі Y
print("\n--- 2. Rotation Y ---")
X_rot, A_rot = rotate_y_3d(model_3d, 45)
print("Матриця:\n", A_rot)
plot_3d(model_3d, X_rot, "Rotation Y (45 deg)")

# 3. Зсув (Shear)
print("\n--- 3. Shear Z ---")
X_sh, A_sh = shear_z_3d(model_3d, 0.5, 0.3)
print("Матриця:\n", A_sh)
plot_3d(model_3d, X_sh, "Shear Z (0.5, 0.3)")

# 4. Відбиття XY
print("\n--- 4. Reflection XY ---")
X_ref, A_ref = reflection_xy_3d(model_3d)
print("Матриця:\n", A_ref)
plot_3d(model_3d, X_ref, "Reflection XY Plane")

# 5. Перевірка некомутативності (Rx @ Ry != Ry @ Rx)
print("\n--- 5. Перевірка некомутативності в 3D ---")
_, Rx = rotate_x_3d(model_3d, 60)
_, Ry = rotate_y_3d(model_3d, 45)

A1 = Rx @ Ry  # Спочатку Y, потім X
A2 = Ry @ Rx  # Спочатку X, потім Y

print("Матриця Rx @ Ry:\n", A1)
print("\nМатриця Ry @ Rx:\n", A2)
print(f"\nЧи однаковий результат? {np.allclose(A1, A2)}")

# Візуалізація різниці некомутативності
X1 = A1 @ model_3d
X2 = A2 @ model_3d

fig = plt.figure(figsize=(12, 6))
all_points = np.hstack([X1, X2])
max_range = np.array([
    all_points[0, :].max() - all_points[0, :].min(),
    all_points[1, :].max() - all_points[1, :].min(),
    all_points[2, :].max() - all_points[2, :].min()
]).max() / 2.0

mid_x = (all_points[0, :].max() + all_points[0, :].min()) * 0.5
mid_y = (all_points[1, :].max() + all_points[1, :].min()) * 0.5
mid_z = (all_points[2, :].max() + all_points[2, :].min()) * 0.5

ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(X1[0, :], X1[1, :], X1[2, :], color='blue', s=2, alpha=0.5)
ax1.set_title("Rx (60 deg) @ Ry (45 deg)")
ax1.set_xlim(mid_x - max_range, mid_x + max_range)
ax1.set_ylim(mid_y - max_range, mid_y + max_range)
ax1.set_zlim(mid_z - max_range, mid_z + max_range)

ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(X2[0, :], X2[1, :], X2[2, :], color='red', s=2, alpha=0.5)
ax2.set_title("Ry (45 deg) @ Rx (60 deg)")
ax2.set_xlim(mid_x - max_range, mid_x + max_range)
ax2.set_ylim(mid_y - max_range, mid_y + max_range)
ax2.set_zlim(mid_z - max_range, mid_z + max_range)

plt.tight_layout()
plt.show()