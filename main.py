from alphabet import ALPHABET_LIST, ALPHABET
import numpy as np

# Macierz P (26x35) — każda litera spłaszczona do wektora
patterns = np.stack([letter.flatten() for letter in ALPHABET_LIST])

# Macierz T (26x26) — identity
targets = np.eye(26)

# Dostęp po nazwie
print(ALPHABET["W"])