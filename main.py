from utils import *
from models import SigmoidNetwork, TanhNetwork, ReLUNetwork

if __name__ == "__main__":
    base_file = "test.train"

    print("\n1. Зчитування згенерованих датасетів")
    X_train, y_train = load_data("train_augmented.txt")
    X_test, _ = load_data("test_generated.txt")

    print("\n2. Порівняння архітектур")

    print("\nТест 1: Класичний Перцептрон (Вхід -> Вихід) + SGD")
    nn_no_hidden = SigmoidNetwork(layer_sizes=[36, 3], loss_func="mse", optimizer="sgd")
    nn_no_hidden.train(X_train, y_train, epochs=1000, learning_rate=0.5)

    # Мало нейронів у прихованому шарі + ReLU + Adam
    print("Тест 2: 5 нейронів у прихованому шарі + ReLU + Adam")
    nn_few_neurons = ReLUNetwork(layer_sizes=[36, 5, 3], loss_func="mse", optimizer="adam")
    nn_few_neurons.train(X_train, y_train, epochs=400, learning_rate=0.01)

    print("Тест 3: 15 нейронів у прихованому шарі + Sigmoid + RMSProp")
    nn_rmsprop = SigmoidNetwork(layer_sizes=[36, 15, 3], loss_func="bce", optimizer="rmsprop")
    nn_rmsprop.train(X_train, y_train, epochs=500, learning_rate=0.01)

    # Багато нейронів у прихованому шарі + Sigmoid + BCE
    print("Тест 4: 30 нейронів у прихованому шарі + Sigmoid + BCE")
    nn_many_neurons = SigmoidNetwork(layer_sizes=[36, 30, 3], loss_func="bce", optimizer="sgd")
    nn_many_neurons.train(X_train, y_train, epochs=1000, learning_rate=0.5)

    # 2 приховані шари + Tanh + Adam + BCE
    print("Тест 5: Глибока мережа + Tanh + Adam + BCE")
    nn_deep = TanhNetwork(layer_sizes=[36, 20, 10, 3], loss_func="bce", optimizer="adam")
    nn_deep.train(X_train, y_train, epochs=400, learning_rate=0.01)

    print("3. Тестування найкращої мережі на нових даних")
    print("\nПрогнози глибокої мережі на 25 тестових образах:")

    predictions = nn_deep.predict(X_test)
    expected_codes = ["а(0,0,0)", "б(0,0,1)", "с(0,1,0)", "д(0,1,1)", "е(1,0,0)"]

    for i, pred in enumerate(predictions):
        expected_letter = expected_codes[i // 5]
        print(f"Образ {i + 1:2d} (Очікується {expected_letter}): Прогноз {pred.tolist()}")
