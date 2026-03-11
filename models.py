import numpy as np
from abc import ABC, abstractmethod


# Базова нейронна мережа (абстрактний клас)
class BaseNeuralNetwork(ABC):
    def __init__(self, layer_sizes, loss_func="bce", optimizer="adam"):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.loss_func = loss_func.lower()  # Функція похибки: mse (mean squared error) | bce (binary cross entropy)
        self.optimizer = optimizer.lower()  # Алгоритм оптимізації: sgd (stochastic gradient descent) | adam
        np.random.seed(42)

        self.weights = [np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.1
                        for i in range(self.num_layers - 1)]  # Ваги
        self.biases = [np.zeros((1, layer_sizes[i + 1]))
                       for i in range(self.num_layers - 1)]  # Зміщення (Bias)

        self.layer_outputs = []
        self.weighted_sums = []
        self.predicted_output = None

        # Параметри для оптимізаторів
        self.m_w, self.v_w = [], []
        self.m_b, self.v_b = [], []
        self.t = 0

        self._init_optimizer_params()

    # Пам'ять для Adam або RMSProp
    def _init_optimizer_params(self):
        if self.optimizer in ["adam", "rmsprop"]:
            self.v_w = [np.zeros_like(w) for w in self.weights]
            self.v_b = [np.zeros_like(b) for b in self.biases]

        if self.optimizer == "adam":
            self.m_w = [np.zeros_like(w) for w in self.weights]
            self.m_b = [np.zeros_like(b) for b in self.biases]

    # Функція активації (прихований шар)
    @abstractmethod
    def activation(self, z):
        pass

    # Похідна функції активації (прихований шар)
    @abstractmethod
    def activation_derivative(self, z):
        pass

    # Функція активації (вихід)
    @staticmethod
    def output_activation(z):
        z_clipped = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z_clipped))

    # Похідна функції активації (вихід)
    def output_activation_derivative(self, z):
        a = self.output_activation(z)
        return a * (1 - a)

    # Розрахунок втрат (функція похибки)
    def compute_loss(self, y_true, y_pred):
        if self.loss_func == "mse":
            return np.square(np.subtract(y_true, y_pred)).mean()
        elif self.loss_func == "bce":
            y_pred_clipped = np.clip(y_pred, 1e-10, 1 - 1e-10)  # Уникнення log(0)
            return -np.mean(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
        return None

    # Розрахунок помилки вихідного шару
    def compute_output_error(self, y_true, y_pred, z_last):
        if self.loss_func == "mse":
            return (y_pred - y_true) * self.output_activation_derivative(z_last)
        return y_pred - y_true

    # Пряме поширення
    def forward(self, x):
        self.layer_outputs = [x]  # Виходи кожного шару
        self.weighted_sums = []  # Зважені суми до застосування функції активації
        current_a = x

        for i in range(self.num_layers - 1):
            z = np.dot(current_a, self.weights[i]) + self.biases[i]
            self.weighted_sums.append(z)

            # Функція активації: сигмоїда для виходу, інші для прихованих
            current_a = self.output_activation(z) if i == self.num_layers - 2 else self.activation(z)
            self.layer_outputs.append(current_a)

        self.predicted_output = self.layer_outputs[-1]
        return self.predicted_output

    # Зворотне поширення помилки
    def backward(self, x_in, y_true, learning_rate):
        m = y_true.shape[0]
        current_error = self.compute_output_error(y_true, self.predicted_output, self.weighted_sums[-1])

        for i in range(self.num_layers - 2, -1, -1):
            a_prev = self.layer_outputs[i]
            dw = np.dot(a_prev.T, current_error) / m
            db = np.sum(current_error, axis=0, keepdims=True) / m

            if i > 0:
                current_error = np.dot(current_error, self.weights[i].T) * self.activation_derivative(
                    self.weighted_sums[i - 1])

            self._update_params(i, dw, db, learning_rate)

    # Оновлення ваг та зміщення відповідно до обраного оптимізатора
    def _update_params(self, i, dw, db, lr):
        eps = 1e-8
        if self.optimizer == "sgd":
            self.weights[i] -= lr * dw
            self.biases[i] -= lr * db

        elif self.optimizer == "rmsprop":
            beta = 0.9
            self.v_w[i] = beta * self.v_w[i] + (1 - beta) * (dw ** 2)
            self.v_b[i] = beta * self.v_b[i] + (1 - beta) * (db ** 2)
            self.weights[i] -= lr * dw / (np.sqrt(self.v_w[i]) + eps)
            self.biases[i] -= lr * db / (np.sqrt(self.v_b[i]) + eps)

        elif self.optimizer == "adam":
            self.t += 1
            b1, b2 = 0.9, 0.999
            # Для ваг
            self.m_w[i] = b1 * self.m_w[i] + (1 - b1) * dw
            self.v_w[i] = b2 * self.v_w[i] + (1 - b2) * (dw ** 2)
            # Корекція зміщення
            m_hat = self.m_w[i] / (1 - b1 ** self.t)
            v_hat = self.v_w[i] / (1 - b2 ** self.t)
            # Фінальні ваги
            self.weights[i] -= lr * m_hat / (np.sqrt(v_hat) + eps)
            # Для зсувів
            self.m_b[i] = b1 * self.m_b[i] + (1 - b1) * db
            self.v_b[i] = b2 * self.v_b[i] + (1 - b2) * (db ** 2)
            # Корекція зміщення
            mb_hat = self.m_b[i] / (1 - b1 ** self.t)
            vb_hat = self.v_b[i] / (1 - b2 ** self.t)
            # Фінальне зміщення
            self.biases[i] -= lr * mb_hat / (np.sqrt(vb_hat) + eps)

    def train(self, x_train, y_train, epochs=5000, learning_rate=0.5):
        print(
            f"\nMLP {self.layer_sizes} | Оптимізатор: {self.optimizer.upper()} | Похибка: {self.loss_func.upper()} | Коефіцієнт швидкості навчання: {learning_rate}")

        log_interval = 100 if self.optimizer in ["adam", "rmsprop"] else 500
        last_epoch = 0

        for epoch in range(epochs):
            y_pred = self.forward(x_train)
            self.backward(x_train, y_train, learning_rate)
            last_epoch = epoch

            if epoch % log_interval == 0:
                current_loss = self.compute_loss(y_train, y_pred)
                print(f"Епоха {epoch:5d} | Похибка: {current_loss:.6f}")

                preds = np.round(y_pred).astype(int)
                acc = (np.sum(np.all(preds == y_train, axis=1)) / y_train.shape[0]) * 100

                if acc == 100.0 and current_loss < 0.01:
                    print("Навчання зупинено достроково! Мережа все вивчила.")
                    break

        y_pred_final = self.forward(x_train)
        final_loss = self.compute_loss(y_train, y_pred_final)

        final_preds = np.round(y_pred_final).astype(int)
        final_accuracy = (np.sum(np.all(final_preds == y_train, axis=1)) / y_train.shape[0]) * 100
        print(
            f"Навчання завершено на епосі {last_epoch + 1}. Фінальна похибка: {final_loss:.6f} | Точність: {final_accuracy:.1f}%\n")

    def predict(self, x):
        return np.round(self.forward(x)).astype(int)


# Сигмоїда
class SigmoidNetwork(BaseNeuralNetwork):
    def activation(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def activation_derivative(self, z):
        a = self.activation(z)
        return a * (1 - a)


# ReLU
class ReLUNetwork(BaseNeuralNetwork):
    def activation(self, z):
        return np.maximum(0, z)

    def activation_derivative(self, z):
        return (z > 0).astype(float)


# Гіперболічний тангенс
class TanhNetwork(BaseNeuralNetwork):
    def activation(self, z):
        return np.tanh(z)

    def activation_derivative(self, z):
        a = self.activation(z)
        return 1 - np.square(a)
