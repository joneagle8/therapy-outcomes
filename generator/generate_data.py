"""
Generates a synthetic therapy-outcomes dataset in the style of NHS Talking
Therapies data.

Real patient-level data isn't publicly available, so this simulates it. The
predictors and outcome rates are based on Kanso et al. (2026), "Building a
prediction model for outcomes following treatment in UK NHS Talking Therapies",
medRxiv v2: https://www.medrxiv.org/content/10.64898/2026.03.12.26348223v2

Predictors of worse outcomes used here, roughly in the paper's order of
importance: unemployment, limited English, benefits, psychotropic
medication, disability/long-term condition, being Muslim, and baseline
severity. The paper ranks these but doesn't give a direction for each, so
directions follow the earlier research it cites (worse outcomes for
minority groups). Ethnicity gets a small effect. The paper calls it a key
predictor but doesn't rank it.

The paper models outcomes, not dropout. Direct effects on dropping out
come from Burke et al. (2025), "Recovery, completion and further referral
after IAPT in Norfolk and Waveney", BJPsych Open, doi:10.1192/bjo.2025.10045:
younger age, being male and living in a more deprived area all lower the
odds of completing treatment.

Ethnicity also gets a small direct dropout effect, extending Kanso's finding
that it predicts outcomes. Burke found no ethnicity effect on completion,
but compared only White British with everyone else, in a mostly White
British area. I keep the effect small for two reasons. Burke's "not
completed" includes people who declined before treatment started. And
Harwood et al. (2023) found the ethnic gaps are mainly at access stages
(self-referral, assessment, starting treatment), which the people modelled
here have already got past. Religion and language affect dropout only
through outcomes.

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

# Chance of responding to therapy, scaled by (1 - risk). Tuned so the
# per-measure outcome rates land near the paper's.
RESPOND_PHQ9 = 2.35
RESPOND_GAD7 = 2.5
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
    "limited_english": 0.40,
    "muslim": 0.25,
    "ethnic_minority": 0.15,
    "early_disengage": 1.1,
    "age_young": 0.25,       # under 25
    "deprivation": 0.05,     # per IMD decile from the middle, most deprived worse
}

# Direct effects on disengaging (log-odds), roughly scaled from Burke et al.
# completion odds ratios: 18-24 ~0.5, 25-35 ~0.7, male ~0.8, and a steady
# gradient of ~2.5x across deprivation deciles.
DISENGAGE = {
    "intercept": -2.1,
    "early_disengage": 1.6,
    "risk": 1.2,
    "under_25": 0.65,
    "age_25_35": 0.3,
    "male": 0.25,
    "deprivation": 0.1,      # per decile from the middle
    "ethnic_minority": 0.15, # small, see docstring
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
    # Religion and language depend on ethnicity. Overall about 10% Muslim
    # and 4-5% with limited English.
    p_muslim = pd.Series(ethnicity).map(
        {"White": 0.03, "Black": 0.12, "Asian": 0.40, "Mixed": 0.08, "Other": 0.15}).to_numpy()
    muslim = rng.random(n_patients) < p_muslim
    other_religion = rng.choice(["Atheist/Agnostic", "Christian", "Other"],
                                size=n_patients, p=[0.50, 0.43, 0.07])
    religion = np.where(muslim, "Muslim", other_religion)
    p_limited_english = pd.Series(ethnicity).map(
        {"White": 0.01, "Black": 0.06, "Asian": 0.15, "Mixed": 0.02, "Other": 0.15}).to_numpy()
    limited_english = rng.random(n_patients) < p_limited_english

    # Deprivation (IMD decile, 1 = most deprived). Referrals skew towards
    # more deprived areas, and unemployment and benefits are more common there.
    imd_decile = rng.choice(np.arange(1, 11), size=n_patients, p=np.linspace(1.4, 0.6, 10) / 10)
    dep_factor = np.interp(imd_decile, [1, 10], [1.5, 0.5])
    dep_centred = 5.5 - imd_decile   # positive = more deprived than average

    # Risk factors
    unemployed = rng.random(n_patients) < 0.22 * dep_factor
    on_benefits = rng.random(n_patients) < 0.18 * dep_factor
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
        + W["limited_english"] * limited_english
        + W["muslim"] * muslim
        + W["ethnic_minority"] * (ethnicity != "White")
        + W["early_disengage"] * early_disengage
        + W["age_young"] * ((age < 25).astype(float))
        + W["deprivation"] * dep_centred
    )
    p_poor = sigmoid(linpred + rng.normal(0, 0.4, n_patients))

    # Where each person would end up after a full course. People either
    # respond, ending in the low range wherever they started, or don't, and
    # improve only a little. Higher risk makes responding less likely. The
    # two measures share one draw, so people tend to respond on both or
    # neither. Scores move steadily towards this end point over the course,
    # so anyone who leaves early keeps whatever they'd gained so far.
    u = rng.random(n_patients)
    responds_phq9 = u < RESPOND_PHQ9 * (1 - p_poor)
    responds_gad7 = u < RESPOND_GAD7 * (1 - p_poor)
    partial_phq9 = baseline_phq9 * (1 - rng.uniform(0, 0.2, n_patients))
    partial_gad7 = baseline_gad7 * (1 - rng.uniform(0, 0.2, n_patients))
    course_end_phq9 = np.where(responds_phq9,
                               np.minimum(partial_phq9, np.clip(rng.normal(5, 2.5, n_patients), 0, None)),
                               partial_phq9)
    course_end_gad7 = np.where(responds_gad7,
                               np.minimum(partial_gad7, np.clip(rng.normal(5, 2, n_patients), 0, None)),
                               partial_gad7)

    # Two ways to leave early. Disengaging is driven by early engagement,
    # overall risk, and directly by age, sex, deprivation and (slightly)
    # ethnicity. Separately, some people stop once they feel better: if
    # their scores have dropped below caseness before the course ends, they
    # may leave within a session or two. The source data doesn't record
    # which kind of leaver someone is, just as real service data wouldn't.
    D = DISENGAGE
    p_disengage = sigmoid(
        D["intercept"]
        + D["early_disengage"] * early_disengage
        + D["risk"] * (p_poor - 0.5)
        + D["under_25"] * (age < 25)
        + D["age_25_35"] * ((age >= 25) & (age <= 35))
        + D["male"] * (~female)
        + D["deprivation"] * dep_centred
        + D["ethnic_minority"] * (ethnicity != "White")
    )
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
        "imd_decile": imd_decile,
        "limited_english": limited_english,
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
    print(f"Ended at referral:  {patients['referral_terminated'].mean():.1%}  (target 20%)")
    print(f"Dropout:            {treated['dropped_out'].mean():.1%}")

    # Per-measure rates, as reported in the paper (target in brackets)
    print("\n              reliable improvement   recovery       reliable recovery")
    for name, col, case, rc, targets in [
        ("Depression", "phq9", CASENESS_PHQ9, RELIABLE_CHANGE_PHQ9, (47.8, 55.7, 46.9)),
        ("Anxiety", "gad7", CASENESS_GAD7, RELIABLE_CHANGE_GAD7, (57.9, 52.9, 49.2)),
    ]:
        g = treated[treated[f"baseline_{col}"] >= case]
        ri = (g[f"baseline_{col}"] - g[f"final_{col}"]) >= rc
        rec = g[f"final_{col}"] < case
        vals = [ri.mean(), rec.mean(), (ri & rec).mean()]
        print(f"  {name:11s}" + "".join(f"  {100*v:5.1f}% ({t:.1f}%)    " for v, t in zip(vals, targets)))

    print("\nRecovery rate with / without each risk factor:")
    treated = treated.assign(muslim=treated["religion"] == "Muslim",
                             ethnic_minority=treated["ethnicity"] != "White")
    for col in ["unemployed", "limited_english", "on_benefits", "psychotropic_med",
                "disability_ltc", "muslim", "ethnic_minority"]:
        hi = treated[treated[col] == True]["recovery"].mean()
        lo = treated[treated[col] == False]["recovery"].mean()
        print(f"  {col:18s} {hi:.1%} / {lo:.1%}")


if __name__ == "__main__":
    main()
