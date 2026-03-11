import numpy as np
import random
import os


def load_data(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Файл {filename} не знайдено :(")

    with open(filename, 'r') as f:
        lines = f.readlines()

    lines = [line.strip() for line in lines if line.strip()]
    num_samples, input_size, output_size = map(int, lines[0].split())
    X = np.zeros((num_samples, input_size))
    y = np.zeros((num_samples, output_size))

    current_line = 1

    for i in range(num_samples):
        pixels = []
        for _ in range(6):
            row_pixels = [int(p) for p in lines[current_line].replace(',', ' ').split()]
            pixels.extend(row_pixels)
            current_line += 1

        X[i] = np.array(pixels)

        targets = [int(t) for t in lines[current_line].replace(',', ' ').split()]
        y[i] = np.array(targets[:output_size])
        current_line += 1

    return X, y


def generate_noisy_dataset(base_X, base_y, num_samples_per_class, max_noise_pixels=2, filename="dataset.txt",
                           is_test=False):
    total_samples = len(base_X) * num_samples_per_class
    input_size = len(base_X[0])
    output_size = len(base_y[0])

    with open(filename, 'w') as f:
        f.write(f"{total_samples} {input_size} {output_size}\n\n")

        for i in range(len(base_X)):
            original_matrix = base_X[i]
            original_code = base_y[i]

            for _ in range(num_samples_per_class):
                noisy_matrix = original_matrix.copy()
                num_pixels_to_flip = random.randint(1, max_noise_pixels)
                flip_indices = random.sample(range(input_size), num_pixels_to_flip)

                for idx in flip_indices:
                    noisy_matrix[idx] = 1 - noisy_matrix[idx]

                for row in range(6):
                    row_pixels = noisy_matrix[row * 6: (row + 1) * 6]
                    f.write(" ".join(map(str, map(int, row_pixels))) + "\n")

                if is_test:
                    f.write(" ".join(["0"] * output_size) + "\n\n")
                else:
                    f.write(" ".join(map(str, map(int, original_code))) + "\n\n")

    print(f"Файл '{filename}' успішно створено! Згенеровано {total_samples} образів.")
