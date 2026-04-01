import multiprocessing as mp
import backtesting

def main():
    # Включаем multiprocessing Pool для backtesting.py
    # (можно указать контекст spawn — по умолчанию на Windows так и будет)
    backtesting.Pool = mp.Pool

    # Импортируй и запусти твой pipeline (лучше через функции из src/runner.py или из notebook-логики,
    # но важно: импорты должны быть на верхнем уровне, а выполнение — внутри main()).
    from src.runner import run_all  # пример: твоя функция, которая запускает всё
    run_all()

if __name__ == "__main__":
    mp.freeze_support()  # полезно на Windows
    main()