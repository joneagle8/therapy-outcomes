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

# Scales how much people improve over a full course. Tuned so recovery
# (of those at caseness) comes out around 50%.
MULT = 2.7
# Chance that someone who's dropped below caseness stops coming early
P_LEAVE_WHEN_BETTER = 0.25

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

    # How much each person would improve over the full course. Higher risk
    # means less improvement. Scores move steadily towards this over the
    # course, so anyone who leaves early keeps whatever they'd gained so far.
    improve_frac_phq9 = np.clip(rng.normal(0.62, 0.2, n_patients) * (1 - p_poor) * MULT, 0, 1)
    improve_frac_gad7 = np.clip(rng.normal(0.62, 0.2, n_patients) * (1 - p_poor) * MULT, 0, 1)
    course_end_phq9 = baseline_phq9 * (1 - improve_frac_phq9)
    course_end_gad7 = baseline_gad7 * (1 - improve_frac_gad7)

    # Two ways to leave early. Disengaging is driven by early engagement and
    # overall risk. Separately, some people stop once they feel better: if
    # their scores have dropped below caseness before the course ends, they
    # may leave within a session or two. The source data doesn't record
    # which kind of leaver someone is, just as real service data wouldn't.
    p_disengage = sigmoid(-1.65 + 1.6 * early_disengage + 1.2 * (p_poor - 0.5))
    disengaged = rng.random(n_patients) < p_disengage

    attendance = []
    stopped_at = np.zeros(n_patients, dtype=int)
    left_improved = np.zeros(n_patients, dtype=bool)
    for i in range(n_patients):
        n_off = sessions_offered[i]
        if disengaged[i]:
            lo, hi = (0.05, 0.3) if early_disengage[i] else (0.25, 0.85)
            last = int(np.clip(round(n_off * rng.uniform(lo, hi)), 1, n_off - 1))
            p_attend = 0.6 if early_disengage[i] else 0.85
        else:
            last = n_off
            p_attend = 0.65 if early_disengage[i] else 0.92
            # first session at which they'd be below caseness on both measures
            frac = np.arange(n_off) / (n_off - 1)
            phq = baseline_phq9[i] + (course_end_phq9[i] - baseline_phq9[i]) * frac
            gad = baseline_gad7[i] + (course_end_gad7[i] - baseline_gad7[i]) * frac
            below = np.flatnonzero((phq < CASENESS_PHQ9) & (gad < CASENESS_GAD7))
            if len(below) and rng.random() < P_LEAVE_WHEN_BETTER:
                leave = below[0] + 1 + rng.integers(0, 2)
                if leave < n_off:
                    last = leave
                    left_improved[i] = True
        att = np.zeros(n_off, dtype=bool)
        att[:last] = rng.random(last) < p_attend
        att[0] = True
        att[last - 1] = True
        attendance.append(att)
        stopped_at[i] = last
    sessions_attended = np.array([a.sum() for a in attendance])
    dropped_out = disengaged | left_improved

    # Final score = where they'd got to on their trajectory when they stopped
    course_frac = (stopped_at - 1) / (sessions_offered - 1)
    final_phq9 = np.clip((baseline_phq9 + (course_end_phq9 - baseline_phq9) * course_frac).round(),
                         0, PHQ9_MAX).astype(int)
    final_gad7 = np.clip((baseline_gad7 + (course_end_gad7 - baseline_gad7) * course_frac).round(),
                         0, GAD7_MAX).astype(int)

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
    for idx, p in treated.iterrows():
        att = attendance[idx]
        n_off = len(att)
        last = int(np.flatnonzero(att)[-1]) + 1
        base, final = p["baseline_phq9"], p["final_phq9"]
        # Scores trend from baseline at session 1 to final at the last attended
        # session, with some session-to-session noise in between
        for s in range(1, n_off + 1):
            if att[s - 1]:
                frac = (s - 1) / max(last - 1, 1)
                score = base + (final - base) * frac
                if 1 < s < last:
                    score += rng.normal(0, 1.5)
                phq9_s = int(np.clip(round(score), 0, PHQ9_MAX))
            else:
                phq9_s = np.nan
            rows.append({
                "patient_id": int(p["patient_id"]),
                "session_number": s,
                "attended": bool(att[s - 1]),
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
    print(f"Recovery:              {treated[treated['at_caseness']]['recovery'].mean():.1%}  (target 50%, of those at caseness)")
    print(f"Reliable improvement:  {treated['reliable_improvement'].mean():.1%}  (target 60%; runs high)")
    print(f"Dropout:               {treated['dropped_out'].mean():.1%}")
    print("\nRecovery rate with / without each risk factor:")
    for col in ["unemployed", "on_benefits", "disability_ltc", "psychotropic_med"]:
        hi = treated[treated[col] == True]["recovery"].mean()
        lo = treated[treated[col] == False]["recovery"].mean()
        print(f"  {col:18s} {hi:.1%} / {lo:.1%}")


if __name__ == "__main__":
    main()
