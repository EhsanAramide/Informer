# Preserved utilities

All helpers are imported from `informer.utils` submodules. These are the original
Informer helpers; they do not choose IonMPX event splits or layer normalization.

## Calendar features (`timefeatures`)

Install the `time` extra for pandas. `time_features` accepts a DataFrame with a
datetime-valued `date` column. With `timeenc=0`, it writes/overwrites `month`,
`day`, `weekday`, `hour`, and `minute` columns in that same DataFrame. Minute
values are integer quarter-hour bins 0–3. Returned columns depend on `freq`:

| Categorical key | Columns |
|---|---|
| y | none |
| m, w | month |
| d, b | month, day, weekday |
| h | month, day, weekday, hour |
| t | month, day, weekday, hour, minute-bin |

With `timeenc=1`, the frame is not modified. pandas offset strings select these
continuous features, in order:

| Offset example | Columns |
|---|---|
| QE, ME | month |
| W | day-of-month, ISO week |
| D, B | weekday, day-of-month, day-of-year |
| h | hour, weekday, day-of-month, day-of-year |
| 15min | minute, hour, weekday, day-of-month, day-of-year |
| s | second, minute, hour, weekday, day-of-month, day-of-year |

Scales subtract 0.5 after division: seconds/minutes by 59, hours by 23,
weekday by 6; one-based day/month/year/week values subtract 1 first and divide
by 30/11/365/52. Valid calendar values lie in [-0.5, 0.5]. Multiples such as
`15min` select features, not resampling. pandas offset names differ from the
model embedding keys (`t`, `m`, `a`); callers must configure both explicitly.
Use modern pandas offset names; deprecated aliases retain pandas warnings/errors.
No timezone conversion, invalid-date repair, or cadence validation is added.

The annual selector returns an empty feature list; the continuous wrapper then
raises at `vstack`. Categorical annual output is `(T, 0)`. Unsupported categorical
keys raise `KeyError`; invalid pandas strings may raise `ValueError`, recognized
but unsupported offsets raise `RuntimeError`. Other `timeenc` values return None.

## Metrics (`metrics`)

MAE, MSE, RMSE, MAPE, and MSPE reduce over all entries. Percentage metrics are
ratios, not multiplied by 100. `metric` returns those five in that order.
RSE divides residual L2 norm by target deviations from the global target mean.
CORR centers along axis zero and uses the reference denominator
`sqrt(sum(centered_true**2 * centered_pred**2, axis=0))`, then averages the
result along the last axis. This is not the usual Pearson denominator and is
preserved verbatim. Zero denominators, constant targets, and empty arrays can
produce warnings, NaN, or infinity; no clipping or fallback is introduced.

## Scaling and training helpers (`tools`)

`StandardScaler.fit` computes NumPy mean and population standard deviation over
axis zero and returns None. Fit with a two-dimensional NumPy array. Transform
and inverse-transform then support NumPy arrays and floating PyTorch tensors;
tensor statistics are converted to the input dtype/device and input gradients
remain connected. These methods return new values without mutating inputs.
If inverse input width differs from fitted width, the reference uses the last
feature's statistics for broadcasting. This does not validate target identity.
Constant features have zero scale and retain division warnings/non-finite values.
Tensor fitting is not a supported fit-to-tensor-transform workflow: statistics
remain tensors, while transform expects NumPy statistics. This scaler is separate
from IonMPX's training-fitted per-feature/per-layer standardizer.

`EarlyStopping` saves `checkpoint.pth` into an existing string-valued directory
path on first use and qualifying improvements. With delta zero, equal losses
also save and reset the counter. `early_stop` remains true once set. Checkpoints
contain only model state. Non-finite losses are not specially handled.

`adjust_learning_rate` mutates all optimizer groups and prints updates. Type1
halves `args.learning_rate` each epoch starting at epoch one. Type2 uses the fixed
reference schedule (epochs 2, 4, 6, 8, 10, 15, 20); other epochs keep the current
rate. Unknown schedule names retain the reference `UnboundLocalError`.
`dotdict` maps attribute get/set/delete to dictionary operations; missing reads
return None and missing deletes raise KeyError.

## Masks (`masking`)

`TriangularCausalMask(B, L)` exposes Boolean `(B, 1, L, L)` values, true only
above the diagonal. `ProbMask` selects causal rows using each batch/head's query
indices and returns the score tensor's shape. True means blocked. Their `mask`
properties expose the stored tensor directly; callers should avoid mutating it.

## Compatibility changes in Chunk 4

Only maintained copies change. `np.Inf` becomes `np.inf` (same positive infinity).
`DatetimeIndex.week` becomes `isocalendar().week.to_numpy(dtype=np.int64)` (same
ISO week numbers, including year boundaries). Missing timestamps remain outside
the supported calendar input contract. Categorical Series.apply calls omit the
deprecated positional convert_dtype=True argument while retaining scalar
extraction and column mutation. Historical root sources stay frozen.
