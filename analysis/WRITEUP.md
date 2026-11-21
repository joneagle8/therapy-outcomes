# Who leaves talking therapy early, and can we spot them in time?

*Project summary - The full analysis is in
[dropout_model.ipynb](dropout_model.ipynb).*

## The question

Around four in ten people who start NHS talking therapy stop before the end
of their course. A service that could spot who is likely to leave, early
enough to do something about it, could help more people get better. So I
asked two questions:

1. Who is more likely to leave early?
2. Using only what a service knows by the third session, how well can we
   predict it?

## A note on the data

Real patient-level therapy records aren't publicly available, so I built a
realistic synthetic dataset of 5,000 referrals. Its patterns come from
published research on NHS Talking Therapies: Kanso et al. (2026) for what
predicts recovery, and Burke et al. (2025) for what predicts leaving early.
The findings below show what this kind of analysis can and can't tell a
service. They aren't new facts about real patients.

## What I found

**Not everyone who leaves early has failed.** 38% of people who started
treatment left early. But about a third of those had already improved
substantially, and half of that group had recovered. Some people simply stop
coming once they feel better. Counting every early leaver as a failure would
send support to people who don't need it, and would hide the group that
does: the quarter of patients who leave without having improved.

**Missed sessions at the start are the clearest warning sign.** Of the people
who came to only one of their first three sessions, 80% left early. Of
those who came to all three, 30% did. But most of that first group (69%)
had already stopped coming by the third session, so the missed sessions
were the dropout happening rather than a sign of it to come. Among people
who came back after session 3, the gap is much smaller: 37% against 27%.
A flat score in those first sessions is a second warning sign, but only for
leaving without improving. People who leave feeling better tend to improve
quickly at the start.

**Younger people, men, and people in more deprived areas are more likely to
leave.**

| | Left early | Left without improving |
|---|---|---|
| Aged 18-24 | 44% | 31% |
| Aged 36 and over | 34% | 21% |
| Men | 42% | 28% |
| Women | 37% | 24% |
| Most deprived areas | 41% | 28% |
| Least deprived areas | 32% | 19% |

**The social factors that predict poorer recovery matter for leaving without
improving, but the effects are small.** Kanso et al. found that unemployment,
benefits, disability, limited English, religion and ethnicity predict
poorer recovery. In this data, each of them makes someone slightly more
likely to leave without improving (roughly 7 to 14% higher odds).
Most of them barely change how many people leave overall, because people
who are doing worse are also less likely to leave because they feel
better, and the two cancel out. Ethnicity is the exception. Ethnic minority
patients leave early slightly more often overall. That part is an
assumption, extending Kanso et al.'s findings, as the research on
ethnicity and dropout itself isn't settled.

Effects this small are easy to miss. With 4,000 patients the analysis
couldn't reliably detect any of them, and some came out pointing the wrong
way. Only when I generated 60,000 patients did they show up clearly. For a
real service, that means looking at a single year's data for one area could
easily suggest "ethnicity makes no difference" when it does. That's a risk
when deciding where to focus outreach.

**A model can flag risk, but not precisely.** I tested how often the model
ranks the right person as higher risk when comparing someone who left with
someone who didn't. For leaving without improving, it gets this right 76 to
80% of the time. For leaving early in general, about 65%, which is in line
with a published dropout model on real NHS data (Shipman and Ushakova,
2026). But a lot of that comes from the people who had already stopped
coming. For people still attending after session 3, it's right only about
55% of the time for leaving early, not much better than a coin toss. It's
useful for prioritising check-ins, but not reliable enough to make
decisions about individuals. The model worked about equally well across
ethnic, religious, language and deprivation groups, which a real service
would need to check carefully.

## What a service could do with this

- **Follow up quickly after a missed early session.** That is the strongest
  signal and costs little to act on. Often the person has already decided
  to stop, so the sooner the better.
- **Review progress at session three.** Someone who has missed sessions and
  whose scores haven't moved is the person most at risk of leaving without
  help.
- **Record why people end treatment**, especially "feeling better" as a
  reason. Without it, services can't separate good endings from bad ones,
  and dropout figures overstate the problem.
- **Look at several years or areas together before concluding a group is
  doing fine.** Small differences between groups need large numbers to see.

## Limits

- The data is synthetic. Every pattern is there because I built it in,
  sized from published studies where possible and by judgement where not.
- The "left without improving" measure uses each person's final score, which
  a service wouldn't have at session three. It's fine for understanding who
  this group is, but couldn't be used for live prediction as it stands.
- Severity is simplified. In this data, people with the most severe symptoms
  show less reliable improvement than those with moderate symptoms. In real
  data it's usually the other way round.

## Sources

- Kanso et al. (2026). Building a prediction model for outcomes following
  treatment in UK NHS Talking Therapies. medRxiv.
- Burke et al. (2025). Recovery, completion and further referral after
  Improving Access to Psychological Therapies in Norfolk and Waveney.
  BJPsych Open.
- Harwood et al. (2023). Variations by ethnicity in referral and treatment
  pathways for IAPT service users in South London. Psychological Medicine.
- Ryan et al. (2013). Factorial invariance of the Patient Health
  Questionnaire and Generalized Anxiety Disorder Questionnaire. British
  Journal of Clinical Psychology.
- Shipman and Ushakova (2026). Fairness-aware machine learning for
  predicting recovery and dropout in NHS Talking Therapies. BJPsych Open.
