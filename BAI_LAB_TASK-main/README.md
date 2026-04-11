# BAI_LAB_TASK
## Zaawansowane Metody Sztucznej Inteligencji — Materiały Laboratoryjne

Witaj! To repozytorium zawiera materiały do laboratoriów z przedmiotu **Zaawansowane Metody Sztucznej Inteligencji (BAI)**. Znajdziesz tu notebooki Jupyter, gotowe implementacje pomocnicze oraz dane do ćwiczeń.

---

## Struktura repozytorium

```
BAI_LAB_TASKS/
│
├── lab1_perceptron/                  # Laboratorium 1 — Perceptron
│   ├── Lab0_Setup_Guide.ipynb        # Notebook konfiguracyjny (zacznij tutaj!)
│   ├── Lab0.1_Theoretical_Introduction.ipynb  # Wstęp teoretyczny do perceptronu
│   ├── Lab1_Perceptron_MAIN_FILE.ipynb        # Główny notebook do wykonania zadań
│   ├── Lab01_perceptron_as_import.ipynb       # Przykład użycia perceptronu jako modułu
│   ├── perceptron.py                 # Gotowa implementacja klasy Perceptron
│   ├── perceptron_task2.py           # Skrypt pomocniczy do zadania 2
│   ├── pdr.py                        # Funkcja pomocnicza: rysowanie granic decyzyjnych
│   ├── data/                         # Dane do ćwiczeń
│   ├── images/                       # Obrazy generowane przez notebooki
│   └── prompts/                      # Podpowiedzi / prompty pomocnicze
│
├── lab2_perceptron/                  # Laboratorium 2 — Perceptron (rozszerzenie)
│   └── data/                         # Dane do ćwiczeń
│
├── Lab2_Perceptron_for_img_recognision_MAIN_FILE.ipynb  # Główny notebook Lab 2
│
└── img/                              # Obrazy współdzielone (wykresy, diagramy)
```

---

## Jak zacząć?

### Krok 1 — Skonfiguruj środowisko

Otwórz i uruchom notebook **`lab1_perceptron/Lab0_Setup_Guide.ipynb`**.

Przeprowadzi Cię przez:
- instalację Pythona (Miniconda),
- tworzenie środowiska `base_ai`,
- instalację wymaganych bibliotek (`numpy`, `pandas`, `matplotlib`, `scikit-learn`),
- test środowiska.

### Krok 2 — Zapoznaj się z teorią

Przeczytaj notebook **`lab1_perceptron/Lab0.1_Theoretical_Introduction.ipynb`**.

Dowiesz się:
- czym jest perceptron i jak działa,
- dlaczego używamy `np.dot` zamiast pętli `for`,
- jak wygląda algorytm uczenia (aktualizacja wag).

### Krok 3 — Wykonaj zadania laboratoryjne

Otwórz główny notebook danego laboratorium i wypełnij zadania:

| Laboratorium | Plik główny | Temat |
|---|---|---|
| Lab 1 | `lab1_perceptron/Lab1_Perceptron_MAIN_FILE.ipynb` | Implementacja i trening perceptronu |
| Lab 2 | `Lab2_Perceptron_for_img_recognision_MAIN_FILE.ipynb` | Perceptron do rozpoznawania obrazów |

---

## Jak pracować z notebookami Jupyter?

1. **Uruchom Jupyter** w terminalu (przy aktywnym środowisku `base_ai`):
   ```bash
   conda activate base_ai
   jupyter notebook
   ```

2. **Nawiguj** do folderu z notebookiem w przeglądarce i otwórz plik `.ipynb`.

3. **Uruchamiaj komórki** kolejno od góry — naciśnij `Shift + Enter` aby wykonać komórkę i przejść do następnej.

4. **Szukaj znacznika `# TWÓJ KOD TUTAJ`** (lub podobnego) — to miejsca, gdzie musisz wpisać swoje rozwiązanie.

5. **Nie usuwaj istniejącego kodu** w komórkach, jeżeli nie masz pewności co robisz — komentarze i pomocnicze fragmenty są tam celowo.

---

## Pliki pomocnicze (nie modyfikuj)

| Plik | Opis |
|---|---|
| `perceptron.py` | Gotowa implementacja klasy `Perceptron` z metodami `fit()` i `predict()` |
| `pdr.py` | Funkcja `plot_decision_regions()` do wizualizacji granic decyzyjnych |

Możesz (i powinieneś!) **czytać** te pliki, żeby zrozumieć jak działają — ale nie musisz ich modyfikować do wykonania zadań.

---

## Środowisko i wymagane biblioteki

```bash
conda create -n base_ai python=3.12 -y
conda activate base_ai
pip install numpy pandas matplotlib scikit-learn notebook ipykernel
python -m ipykernel install --user --name base_ai --display-name "Python (base_ai)"
```

---

## Masz pytania?

Jeśli napotkasz na problemy z konfiguracją lub zrozumieniem materiału:
1. Sprawdź najpierw **`Lab0_Setup_Guide.ipynb`** — odpowiada na najczęstsze problemy środowiskowe.
2. Zajrzyj do **`Lab0.1_Theoretical_Introduction.ipynb`** po wyjaśnienia teoretyczne.
3. Skontaktuj się z prowadzącym zajęcia.
