import argparse, sys
from pathlib import Path
import pandas as pd

from src.lab2_starter   import load_predictions, make_splits, save_run, split_summary, train_once
from src.threshold      import threshold_search_fast, verify_algorithms
from src.toy_verifivation import run_toy_verifications
from src.analysis       import plot_pr_roc, compare_three_thresholds, test_metrics, error_table


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--data",  type=Path, help="Шлях до creditcard.csv")
    mode.add_argument("--saved", type=Path, help="Тека із збереженими прогнозами")
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()

    if args.data:
        out = args.out
        out.mkdir(parents=True, exist_ok=True)
        print(f"Навчання на {args.data} ...")
        train, val, test, scaler = make_splits(args.data)
        model = train_once(train)
        saved = save_run(out, model, (train, val, test), scaler, args.data)
    else:
        out = args.saved
        if not out.exists():
            sys.exit(f"Тека {out} не знайдена.")
        saved = load_predictions(out)
        print(f"Завантажено прогнози з {out}")

    print("\n=== 1. Розподіл (train / validation / test) ===")
    print(split_summary(saved).to_string(index=False))

    y_val, s_val = saved["validation_y"], saved["validation_scores"]
    print("\n=== 2. Пошук порогу (Validation) ===")
    best_t, candidates = threshold_search_fast(y_val, s_val)
    candidates.to_csv(out / "validation_candidates.csv", index=False)
    print(f"Збережено таблицю кандидатів -> {out}/validation_candidates.csv")

    auc = plot_pr_roc(y_val, s_val, out)
    print(f"ROC-AUC = {auc:.6f} | t* = {best_t:.6f}")
    print(compare_three_thresholds(y_val, s_val, best_t).to_string(index=False))

    print("\nВерифікація fast vs slow на validation ...")
    verify_algorithms(y_val, s_val, label="validation")

    print("\n=== 3. Перевірка на наборах A, B, C ===")
    run_toy_verifications()

    y_te, s_te = saved["test_y"], saved["test_scores"]
    print(f"=== 4. Оцінка на Test (t* = {best_t:.6f}) ===")
    m = test_metrics(y_te, s_te, best_t)
    for k, v in m.items():
        print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

    df_err = error_table(y_te, s_te, saved["test_amount"], saved["test_row_ids"], best_t)
    print("\nПерші 5 FP та 5 FN на тесті:")
    print(df_err.to_string(index=False))
    df_err.to_csv(out / "test_errors.csv", index=False)
    print(f"\nГОТОВО. Результати збережено в {out}/")


if __name__ == "__main__":
    main()
