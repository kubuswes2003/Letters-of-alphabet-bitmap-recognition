import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import logging

from alphabet import ALPHABET_LIST

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Parametry sieci i treningu
NUM_LETTERS = 26
PIXEL_COUNT = 35        # bitmapa 5x7 = 35 pikseli wejściowych
BITMAP_ROWS = 7
BITMAP_COLS = 5
MAX_EPOCHS = 500
MSE_THRESHOLD = 1e-5    # warunek stopu, kończymy jeśli błąd spadnie poniżej
LEARNING_RATE = 0.01
HIDDEN_NEURONS = 30     # liczba neuronów w warstwie ukrytej
NOISE_LEVELS = [0.1, 0.2, 0.5]
LETTER_NAMES = [chr(ord("A") + i) for i in range(NUM_LETTERS)]


# Definicja sieci: wejście (35) -> ukryta (30) -> wyjście (26)
class LetterRecognizer(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        # Warstwy sieci: wejście -> ukryta -> wyjście
        self.hidden = nn.Linear(input_size, hidden_size)
        self.activation = nn.Tanh()     # odpowiednik tansig z MATLABa
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.hidden(x)
        x = self.activation(x)
        x = self.output(x)      # brak aktywacji na wyjściu = purelin
        return x


# Trening sieci: zwraca historię MSE żeby później narysować wykres
def train_model(
    model: LetterRecognizer,
    patterns: torch.Tensor,
    targets: torch.Tensor,
) -> list[float]:
    criterion = nn.MSELoss()
    # optymizer adaptujący learning rate, działa lepiej niż zwykłe SGD
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_history = []

    for epoch in range(MAX_EPOCHS):
        output = model(patterns)
        loss = criterion(output, targets)
        loss_history.append(loss.item())

        # Wczesne zatrzymanie gdy błąd jest wystarczająco mały
        if loss.item() < MSE_THRESHOLD:
            logger.info("Trening zakończony w epoce %d, MSE = %.6f", epoch, loss.item())
            break

        # pętla treningowa: zeruj gradienty -> backprop -> aktualizuj wagi
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            logger.info("Epoka %d, MSE = %.6f", epoch, loss.item())
    else:
        logger.info("Osiągnięto max epok (%d), MSE = %.6f", MAX_EPOCHS, loss.item())

    return loss_history


# Test na czystych danych: sprawdza czy sieć rozpoznaje wszystkie 26 liter
def test_clean(
    model: LetterRecognizer,
    patterns: torch.Tensor,
) -> list[bool]:
    results = []
    with torch.no_grad():   # przy testowaniu nie liczymy gradientów
        for i in range(NUM_LETTERS):
            output = model(patterns[i].unsqueeze(0))    # unsqueeze dodaje wymiar batcha
            predicted = torch.argmax(output).item()     # neuron z najwyższą wartością = predykcja
            is_correct = predicted == i
            results.append(is_correct)

            status = "OK" if is_correct else f"BŁĄD -> {LETTER_NAMES[predicted]}"
            logger.info("Litera %s: %s", LETTER_NAMES[i], status)

    accuracy = sum(results) / NUM_LETTERS * 100
    logger.info("Accuracy (czyste dane): %.1f%%", accuracy)
    return results


# Test z szumem gaussowskim: zwraca wyniki, zaszumione bitmapy i predykcje
def test_noisy(
    model: LetterRecognizer,
    noise_level: float,
) -> tuple[list[bool], list[np.ndarray], list[int]]:
    results = []
    noisy_bitmaps = []
    predictions = []

    with torch.no_grad():
        for i in range(NUM_LETTERS):
            clean = ALPHABET_LIST[i].flatten().astype(np.float64)

            # Dodajemy szum gaussowski i obcinamy wartości do [0, 1]
            noisy = np.clip(clean + np.random.randn(PIXEL_COUNT) * noise_level, 0.0, 1.0)
            noisy_bitmaps.append(noisy.reshape(BITMAP_ROWS, BITMAP_COLS))

            noisy_tensor = torch.tensor(noisy, dtype=torch.float32).unsqueeze(0)
            output = model(noisy_tensor)
            predicted = torch.argmax(output).item()

            predictions.append(predicted)
            results.append(predicted == i)

    accuracy = sum(results) / NUM_LETTERS * 100
    logger.info("Noise %.1f: accuracy = %.1f%%", noise_level, accuracy)
    return results, noisy_bitmaps, predictions


# Wykres MSE w czasie treningu
def plot_training(loss_history: list[float]) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(loss_history, color="#2563eb", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Training Loss Over Epochs")
    ax.set_yscale("log")    # skala logarytmiczna: łatwiej widać spadek na początku
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("plot_training.png", dpi=150)
    plt.close(fig)
    logger.info("Saved: plot_training.png")


# Siatka wszystkich 26 liter z zaznaczeniem poprawnych/błędnych rozpoznań
def plot_clean_results(results: list[bool]) -> None:
    fig, axes = plt.subplots(2, 13, figsize=(16, 4))
    fig.suptitle("Clean Pattern Recognition Results", fontsize=14, fontweight="bold")

    for i in range(NUM_LETTERS):
        row, col = divmod(i, 13)
        ax = axes[row, col]
        ax.imshow(ALPHABET_LIST[i], cmap="Blues", vmin=0, vmax=1, aspect="equal")
        # Zielony tytuł = poprawnie rozpoznana, czerwony = błąd
        color = "#16a34a" if results[i] else "#dc2626"
        ax.set_title(LETTER_NAMES[i], fontsize=12, fontweight="bold", color=color)
        ax.axis("off")

    fig.tight_layout()
    fig.savefig("plot_clean.png", dpi=150)
    plt.close(fig)
    logger.info("Saved: plot_clean.png")


# Porównanie czystych i zaszumionych bitmap z predykcją sieci
def plot_noisy_results(
    noise_level: float,
    results: list[bool],
    noisy_bitmaps: list[np.ndarray],
    predictions: list[int],
) -> None:
    fig, axes = plt.subplots(4, 13, figsize=(16, 8))
    fig.suptitle(
        f"Noisy Recognition noise = {noise_level}",
        fontsize=14,
        fontweight="bold",
    )

    for i in range(NUM_LETTERS):
        col = i % 13
        # Litery A-M w wierszach 0-1, N-Z w wierszach 2-3
        row_clean = 0 if i < 13 else 2
        row_noisy = row_clean + 1

        # Górny wiersz: oryginalna bitmapa
        ax_clean = axes[row_clean, col]
        ax_clean.imshow(ALPHABET_LIST[i], cmap="Blues", vmin=0, vmax=1, aspect="equal")
        ax_clean.set_title(LETTER_NAMES[i], fontsize=10, fontweight="bold")
        ax_clean.axis("off")

        # Dolny wiersz: zaszumiona bitmapa + predykcja sieci
        ax_noisy = axes[row_noisy, col]
        ax_noisy.imshow(noisy_bitmaps[i], cmap="Blues", vmin=0, vmax=1, aspect="equal")
        pred_name = LETTER_NAMES[predictions[i]]
        color = "#16a34a" if results[i] else "#dc2626"
        ax_noisy.set_title(f"→{pred_name}", fontsize=10, fontweight="bold", color=color)
        ax_noisy.axis("off")

    fig.tight_layout()
    filename = f"plot_noisy_{str(noise_level).replace('.', '_')}.png"
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", filename)


# Wizualizacja wag warstwy ukrytej: pokazuje na co każdy neuron "patrzy"
def plot_neuron_weights(model: LetterRecognizer) -> None:
    # Wagi warstwy ukrytej: kształt (30, 35) 30 neuronów, każdy ma 35 wag
    weights = model.hidden.weight.detach().numpy()

    cols = 10
    rows = HIDDEN_NEURONS // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, 6))
    fig.suptitle(
        "Hidden Neuron Weight Filters\n"
        "(blue = pixel activates neuron, red = pixel suppresses neuron)",
        fontsize=12,
        fontweight="bold",
    )

    for idx in range(HIDDEN_NEURONS):
        row, col = divmod(idx, cols)
        ax = axes[row, col]
        # Reshapujemy wektor 35 wag z powrotem do kształtu bitmapy 7x5
        # dzięki temu widzimy przestrzennie na co dany neuron "patrzy"
        weight_map = weights[idx].reshape(BITMAP_ROWS, BITMAP_COLS)
        ax.imshow(weight_map, cmap="RdBu", aspect="equal")
        ax.set_title(f"N{idx}", fontsize=8)
        ax.axis("off")

    fig.tight_layout()
    fig.savefig("plot_neuron_weights.png", dpi=150)
    plt.close(fig)
    logger.info("Saved: plot_neuron_weights.png")


# Heatmapa aktywacji neuronów ukrytych dla każdej litery
def plot_hidden_activations(model: LetterRecognizer, patterns: torch.Tensor) -> None:
    # Przepuszczamy dane tylko przez warstwę ukrytą (bez warstwy wyjściowej)
    # żeby zobaczyć reprezentację wewnętrzną sieci
    with torch.no_grad():
        activations = model.activation(model.hidden(patterns))  # kształt: (26, 30)

    act_np = activations.numpy()

    fig, ax = plt.subplots(figsize=(14, 8))
    # Transponujemy żeby neurony były w wierszach, litery w kolumnach
    im = ax.imshow(act_np.T, cmap="RdBu", aspect="auto", vmin=-1, vmax=1)

    ax.set_xticks(range(NUM_LETTERS))
    ax.set_xticklabels(LETTER_NAMES, fontsize=10)
    ax.set_yticks(range(HIDDEN_NEURONS))
    ax.set_yticklabels([f"N{i}" for i in range(HIDDEN_NEURONS)], fontsize=8)
    ax.set_xlabel("Input Letter", fontsize=12)
    ax.set_ylabel("Hidden Neuron", fontsize=12)
    ax.set_title(
        "Hidden Layer Activations per Letter\n"
        "(blue = strongly activated, red = strongly suppressed)",
        fontsize=12,
        fontweight="bold",
    )
    fig.colorbar(im, ax=ax, label="Tanh activation")
    fig.tight_layout()
    fig.savefig("plot_hidden_activations.png", dpi=150)
    plt.close(fig)
    logger.info("Saved: plot_hidden_activations.png")


# Macierz konfuzji: które litery mylą się ze sobą przy danym poziomie szumu
def plot_confusion_matrix(
    model: LetterRecognizer,
    noise_level: float,
    n_trials: int = 20,
) -> None:
    # Macierz 26x26: confusion[i, j] = ile razy litera i została rozpoznana jako j
    confusion = np.zeros((NUM_LETTERS, NUM_LETTERS), dtype=int)

    with torch.no_grad():
        # Testujemy wielokrotnie żeby uśrednić losowość szumu
        for _ in range(n_trials):
            for i in range(NUM_LETTERS):
                clean = ALPHABET_LIST[i].flatten().astype(np.float64)
                noisy = np.clip(clean + np.random.randn(PIXEL_COUNT) * noise_level, 0.0, 1.0)
                noisy_tensor = torch.tensor(noisy, dtype=torch.float32).unsqueeze(0)
                predicted = torch.argmax(model(noisy_tensor)).item()
                confusion[i, predicted] += 1

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(confusion, cmap="Blues")

    ax.set_xticks(range(NUM_LETTERS))
    ax.set_yticks(range(NUM_LETTERS))
    ax.set_xticklabels(LETTER_NAMES, fontsize=9)
    ax.set_yticklabels(LETTER_NAMES, fontsize=9)
    ax.set_xlabel("Predicted Letter", fontsize=12)
    ax.set_ylabel("True Letter", fontsize=12)
    ax.set_title(
        f"Confusion Matrix  noise = {noise_level}  ({n_trials} trials per letter)",
        fontsize=12,
        fontweight="bold",
    )

    # Wartości w komórkach: pomijamy zera żeby nie zaśmiecać wykresu
    for i in range(NUM_LETTERS):
        for j in range(NUM_LETTERS):
            if confusion[i, j] > 0:
                color = "white" if confusion[i, j] > n_trials * 0.6 else "black"
                ax.text(j, i, str(confusion[i, j]), ha="center", va="center",
                        fontsize=7, color=color)

    fig.colorbar(im, ax=ax, label="Count")
    fig.tight_layout()
    filename = f"plot_confusion_{str(noise_level).replace('.', '_')}.png"
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", filename)


# Wykres słupkowy porównujący accuracy dla różnych poziomów szumu
def plot_accuracy_summary(noise_accuracies: dict[float, float]) -> None:
    labels = ["Clean"] + [f"Noise {n}" for n in noise_accuracies]
    values = [100.0] + list(noise_accuracies.values())
    colors = ["#2563eb"] + ["#f59e0b", "#f97316", "#dc2626"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=colors[:len(values)], edgecolor="white", linewidth=1.5)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{val:.1f}%",
            ha="center",
            fontsize=12,
            fontweight="bold",
        )

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Recognition Accuracy vs Noise Level")
    ax.set_ylim(0, 115)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig("plot_accuracy_summary.png", dpi=150)
    plt.close(fig)
    logger.info("Saved: plot_accuracy_summary.png")


if __name__ == "__main__":
    # Każda litera 7x5 spłaszczona do wektora 35-elementowego, wszystkie sklejone w macierz 26x35
    patterns = torch.tensor(
        np.stack([letter.flatten() for letter in ALPHABET_LIST]),
        dtype=torch.float32,
    )
    # Macierz docelowa: litera i ma 1 na pozycji i, 0 wszędzie indziej
    targets = torch.eye(NUM_LETTERS, dtype=torch.float32)

    # Trening
    model = LetterRecognizer(PIXEL_COUNT, HIDDEN_NEURONS, NUM_LETTERS)
    loss_history = train_model(model, patterns, targets)
    plot_training(loss_history)

    # Test na czystych danych (powinno być 100%)
    model.eval()
    clean_results = test_clean(model, patterns)
    plot_clean_results(clean_results)

    # Test z szumem dla każdego poziomu
    noise_accuracies: dict[float, float] = {}
    for noise_level in NOISE_LEVELS:
        results, noisy_bitmaps, predictions = test_noisy(model, noise_level)
        noise_accuracies[noise_level] = sum(results) / NUM_LETTERS * 100
        plot_noisy_results(noise_level, results, noisy_bitmaps, predictions)

    plot_accuracy_summary(noise_accuracies)

    # Wizualizacja tego czego nauczyła się sieć
    plot_neuron_weights(model)
    plot_hidden_activations(model, patterns)

    # Macierz konfuzji dla dwóch poziomów szumu
    for noise_level in [0.2, 0.5]:
        plot_confusion_matrix(model, noise_level)