# Gate 6 nested meta-init evidence

- status: `not-supported`
- mechanism passed: `True`
- causal passed: `False`
- locked partition consumed once: `True`
- user-related prior supported: `False`

## Locked meta-init vs controls

- `copy-init`: step gain `0.000000`, AUC gain `-0.000171`, final-error delta `-0.000012`, minimum effect `False`
- `random-init`: step gain `4.666667`, AUC gain `0.260020`, final-error delta `-0.004755`, minimum effect `True`
- `no-init`: step gain `4.000000`, AUC gain `0.225409`, final-error delta `-0.005369`, minimum effect `True`

## Claim boundary

- This packet tests only nested initialization. It does not inherit or reverse the Gate 5 CMS Pareto NO-GO.
- The same locked partition may not be tuned against or rerun after a failed preregistered gate.
