"""
Generates a synthetic therapy-outcomes dataset in the style of NHS Talking
Therapies data.

Real patient-level data isn't publicly available, so this simulates it. The
predictors and outcome rates are based on Kanso et al. (2026), "Building a
prediction model for outcomes following treatment in UK NHS Talking Therapies",
medRxiv v2: https://www.medrxiv.org/content/10.64898/2026.03.12.26348223v2

Predictors of worse outcomes used here: higher baseline severity,
unemployment, benefits, disability/long-term condition, psychotropic
medication. Recovery is around 50%.

Writes two files:
  patients.csv  one row per patient
  sessions.csv  one row per scheduled session

Usage:
    python generate_data.py
    python generate_data.py --n 2000 --seed 7 --out ../data
"""

import argparse
import os
import numpy as np
import pandas as pd


# Scale limits and NHS-TT thresholds
PHQ9_MAX = 27          # PHQ-9 scale 0-27
GAD7_MAX = 21          # GAD-7 scale 0-21
CASENESS_PHQ9 = 10     # NHS-TT caseness thresholds
CASENESS_GAD7 = 8
RELIABLE_CHANGE_PHQ9 = 6   # min drop for reliable improvement
RELIABLE_CHANGE_GAD7 = 4

# Log-odds weights for the poor-outcome score. Positive = worse outcome.
W = {
    "intercept": -0.35,
    "baseline_sev": 0.9,     # strongest effect
    "unemployed": 0.45,
    "on_benefits": 0.40,
    "disability_ltc": 0.35,
    "psychotropic_med": 0.30,
    "early_disengage": 1.1,
    "age_young": 0.25,       # under 25
}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def generate(n_patients: int, seed: int):
    rng = np.random.default_rng(seed)

    # Demographics (proportions from the paper's sample)
    age = np.clip(rng.normal(34, 12, n_patients), 18, 80).round().astype(int)
    female = rng.random(n_patients) < 0.73

    ethnicity = rng.choice(
        ["White", "Black", "Asian", "Mixed", "Other"],
        size=n_patients, p=[0.57, 0.22, 0.12, 0.05, 0.04],
    )
    religion = rng.choice(
        ["Atheist/Agnostic", "Christian", "Muslim", "Other"],
        size=n_patients, p=[0.45, 0.38, 0.11, 0.06],
    )

    # Risk factors
    unemployed = rng.random(n_patients) < 0.22
    on_benefits = rng.random(n_patients) < 0.18
    disability_ltc = rng.random(n_patients) < 0.20
    psychotropic_med = rng.random(n_patients) < 0.30

    # Baseline scores. Most referrals are above caseness.
    baseline_phq9 = np.clip(rng.normal(15, 5, n_patients), 0, PHQ9_MAX).round().astype(int)
    baseline_gad7 = np.clip(rng.normal(12, 4, n_patients), 0, GAD7_MAX).round().astype(int)
    baseline_sev_z = (baseline_phq9 - 15) / 5.0

    # About 20% of referrals end at assessment and never start treatment.
    # This is separate from dropping out during treatment.
    referral_terminated = rng.random(n_patients) < 0.20

    sessions_offered = rng.integers(8, 21, n_patients)

    # Poor engagement in the first few sessions
    early_disengage = rng.random(n_patients) < 0.25

    linpred = (
        W["intercept"]
        + W["baseline_sev"] * baseline_sev_z
        + W["unemployed"] * unemployed
        + W["on_benefits"] * on_benefits
        + W["disability_ltc"] * disability_ltc
        + W["psychotropic_med"] * psychotropic_med
        + W["early_disengage"] * early_disengage
        + W["age_young"] * ((age < 25).astype(float))
    )
    p_poor = sigmoid(linpred + rng.normal(0, 0.4, n_patients))

    # Dropout depends mostly on early engagement, partly on overall risk
    p_dropout = sigmoid(-1.0 + 1.6 * early_disengage + 1.2 * (p_poor - 0.5))
    dropped_out = rng.random(n_patients) < p_dropout

    # Dropouts attend fewer of their offered sessions
    attend_fraction = np.where(dropped_out,
                               rng.uniform(0.1, 0.5, n_patients),
                               rng.uniform(0.6, 1.0, n_patients))
    sessions_attended = np.minimum(sessions_offered,
                                   np.ceil(sessions_offered * attend_fraction)).astype(int)
    sessions_attended = np.maximum(sessions_attended, 1)

    # Final scores. Higher risk means less improvement, and dropouts improve
    # less again. The 2.3 multiplier was tuned to get ~50% recovery and ~60%
    # reliable improvement.
    improve_frac_phq9 = np.clip(rng.normal(0.62, 0.2, n_patients) * (1 - p_poor) * 2.3, 0, 1)
    improve_frac_gad7 = np.clip(rng.normal(0.62, 0.2, n_patients) * (1 - p_poor) * 2.3, 0, 1)
    improve_frac_phq9 = np.where(dropped_out, improve_frac_phq9 * 0.4, improve_frac_phq9)
    improve_frac_gad7 = np.where(dropped_out, improve_frac_gad7 * 0.4, improve_frac_gad7)

    final_phq9 = np.clip((baseline_phq9 * (1 - improve_frac_phq9)).round(), 0, PHQ9_MAX).astype(int)
    final_gad7 = np.clip((baseline_gad7 * (1 - improve_frac_gad7)).round(), 0, GAD7_MAX).astype(int)

    # Outcome flags
    at_caseness = (baseline_phq9 >= CASENESS_PHQ9) | (baseline_gad7 >= CASENESS_GAD7)
    reliable_improvement = ((baseline_phq9 - final_phq9) >= RELIABLE_CHANGE_PHQ9) | \
                           ((baseline_gad7 - final_gad7) >= RELIABLE_CHANGE_GAD7)
    recovery = at_caseness & (final_phq9 < CASENESS_PHQ9) & (final_gad7 < CASENESS_GAD7)
    reliable_recovery = recovery & reliable_improvement

    patients = pd.DataFrame({
        "patient_id": np.arange(1, n_patients + 1),
        "age": age,
        "female": female,
        "ethnicity": ethnicity,
        "religion": religion,
        "unemployed": unemployed,
        "on_benefits": on_benefits,
        "disability_ltc": disability_ltc,
        "psychotropic_med": psychotropic_med,
        "baseline_phq9": baseline_phq9,
        "baseline_gad7": baseline_gad7,
        "referral_terminated": referral_terminated,
        "sessions_offered": sessions_offered,
        "sessions_attended": sessions_attended,
        "dropped_out": dropped_out,
        "final_phq9": final_phq9,
        "final_gad7": final_gad7,
        "at_caseness": at_caseness,
        "reliable_improvement": reliable_improvement,
        "recovery": recovery,
        "reliable_recovery": reliable_recovery,
    })

    # No treatment data for patients who ended at referral. Convert to
    # nullable types first so these columns can hold NA.
    treat_cols = ["sessions_attended", "dropped_out", "final_phq9", "final_gad7",
                  "reliable_improvement", "recovery", "reliable_recovery"]
    bool_cols = ["dropped_out", "reliable_improvement", "recovery", "reliable_recovery"]
    int_cols = ["sessions_attended", "final_phq9", "final_gad7"]
    for c in bool_cols:
        patients[c] = patients[c].astype("boolean") 
    for c in int_cols:
        patients[c] = patients[c].astype("Int64")
    patients.loc[patients["referral_terminated"], treat_cols] = pd.NA

    # Sessions table
    rows = []
    treated = patients[~patients["referral_terminated"]]
    for _, p in treated.iterrows():
        n_off = int(p["sessions_offered"])
        n_att = int(p["sessions_attended"])
        attended_idx = set(rng.choice(np.arange(1, n_off + 1), size=n_att, replace=False))
        # Score moves in a straight line from baseline to final
        for s in range(1, n_off + 1):
            attended = s in attended_idx
            frac = (s - 1) / max(n_off - 1, 1)
            phq9_s = round(p["baseline_phq9"] + (p["final_phq9"] - p["baseline_phq9"]) * frac) if attended else np.nan
            rows.append({
                "patient_id": int(p["patient_id"]),
                "session_number": s,
                "attended": attended,
                "phq9_score": phq9_s,
            })
    sessions = pd.DataFrame(rows)

    return patients, sessions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000, help="number of patients")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "data"))
    args = ap.parse_args()

    patients, sessions = generate(args.n, args.seed)
    os.makedirs(args.out, exist_ok=True)
    p_path = os.path.join(args.out, "patients.csv")
    s_path = os.path.join(args.out, "sessions.csv")
    patients.to_csv(p_path, index=False)
    sessions.to_csv(s_path, index=False)

    # Quick check that the rates came out as intended
    treated = patients[~patients["referral_terminated"]]
    print(f"{len(patients)} patients written to {p_path}")
    print(f"{len(sessions)} sessions written to {s_path}")
    print()
    print(f"Ended at referral:     {patients['referral_terminated'].mean():.1%}  (target 20%)")
    print(f"Recovery:              {treated['recovery'].mean():.1%}  (target 50%)")
    print(f"Reliable improvement:  {treated['reliable_improvement'].mean():.1%}  (target 60%)")
    print(f"Dropout:               {treated['dropped_out'].mean():.1%}")
    print("\nRecovery rate with / without each risk factor:")
    for col in ["unemployed", "on_benefits", "disability_ltc", "psychotropic_med"]:
        hi = treated[treated[col] == True]["recovery"].mean()
        lo = treated[treated[col] == False]["recovery"].mean()
        print(f"  {col:18s} {hi:.1%} / {lo:.1%}")


if __name__ == "__main__":
    main()
