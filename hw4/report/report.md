# CS285 HW4 Report: LLM Reinforcement Learning

For this homework, I implemented the required LLM RL pieces and ran the four main experiments with `Qwen/Qwen2.5-Math-1.5B-Instruct`. My main takeaway was that `format_copy` did exactly what a debugging task should do, and both methods solved it quickly. `math_hard` was much more revealing. In my runs, GRPO finished at `0.3984` eval exact match on the boxed-answer metric, while GR-REINFORCE finished at `0.2695`.

## 1. Approximate KL

I found it easiest to think of the KL term here as something we estimate on sampled completion tokens rather than compute exactly over the whole vocabulary at every decoding step. If a token `a` is sampled from the current policy and we define `Δ(a) = log π_ref(a | s) - log π_θ(a | s)`, then `exp(Δ(a)) = π_ref(a | s) / π_θ(a | s)`. Taking expectation under `a ~ π_θ`, we get `E[exp(Δ(a))] = 1`, so `E[exp(Δ(a)) - Δ(a) - 1] = -E[Δ(a)] = E[log π_θ(a | s) - log π_ref(a | s)]`, which is exactly `KL(π_θ || π_ref)`. That is why `exp(Δ) - Δ - 1` is a valid sampled-token estimator for the KL term used here. I also prefer this form to just using `-Δ`, because each sampled term is nonnegative, so it looks more like a local divergence penalty instead of sometimes acting like a reward.

If to compute the exact KL at every token position, it would be much more expensive because we would need to compare the two full next-token distributions over the whole vocabulary, not just the sampled token. In practice that means carrying around and operating on large `[batch, sequence, vocab]` tensors, doing full log-softmax work for both policies, and then reducing over the vocabulary dimension at every position. The sampled-token version is much cheaper because it reuses the log-probabilities we already need for the policy objective and only averages over the generated completion tokens.

## 2. Implementation

I basically followed the order the handout recommends. I started with per-token log-probabilities, completion masking, and the approximate KL computation. After that I implemented minibatching for rollout buffers, then group-relative advantages and optional advantage normalization, and only then moved on to the update rules for GR-REINFORCE and GRPO. Doing it in that order made debugging a lot easier, since both algorithms rely on the same rollout tensors and masking logic.

The main confusion point for me was token alignment. The model outputs logits at position `t - 1`, but those logits are used to score the target token at position `t`, so it is very easy to get an off-by-one bug if you are not careful. The other thing that took a bit of care was making sure I only scored generated completion tokens and not prompt tokens or padding. Once that was fixed, the RL losses started behaving much more sensibly, especially on `format_copy`, where implementation mistakes show up right away.

## 3. GR-REINFORCE vs. GRPO on Math

In my `math_hard` runs, the difference between the two algorithms was pretty clear. Both runs start from the same baseline eval exact match of `0.2266`, so they begin in essentially the same place. By the evaluation around step `199`, GR-REINFORCE reaches `0.2813`, while GRPO is already at `0.3398`. By the end, GR-REINFORCE finishes at `0.2715` and GRPO finishes at `0.3984`. The rollout reward is noisier than the eval curve, which is exactly what the homework warns about, but I still saw the same pattern there: around step `200`, GR-REINFORCE is at `0.4734` reward and GRPO is at `0.5453`.

| Run                          | Start eval exact match | Eval around step 199 | Final eval exact match |
| :--------------------------- | :--------------------: | :------------------: | :--------------------: |
| `math_hard` + GR-REINFORCE |         0.2266         |        0.2813        |         0.2715         |
| `math_hard` + GRPO         |         0.2266         |        0.3398        |         0.3984         |

The two math runs use the same batch size, group size, minibatch size, grad accumulation, and learning rate. The main algorithmic difference is that GR-REINFORCE is strictly single-pass and on-policy, while GRPO reuses the same rollout for two PPO epochs with clipping. So over the first 200 iterations, GRPO gets more optimization work out of each collected batch. That is exactly what the curves look like to me: GRPO learns faster from the same amount of sampled data, while REINFORCE stays much closer to the reference policy and moves more slowly.

One other detail that stood out to me is KL. Near the end of the REINFORCE run, the approximate KL stays tiny, around `0.005`, while GRPO is willing to move farther from the reference policy. My read is that this extra movement really does matter on `math_hard`, where the model seems to need more than a tiny nudge to improve both answer accuracy and formatting behavior.

## 4. GRPO Ablations on `format_copy`

For the extra GRPO study on `format_copy`, I kept the task fixed and changed one part of the update at a time. All of these runs still solved the task in the sense that eval exact match stayed at `1.0`, so I found the more informative signals were rollout reward, KL, clip fraction, and runtime.

```
eval/math_hard_test_subset_split_fraction_exact_match_using_boxed_answer_parser
```


| Variation              | `eval/exact_match` | `rollout/reward` | `approx_kl` | `clip_fraction` | Runtime |
| :--------------------- | :------------------: | :----------------: | :-----------: | :---------------: | :-----: |
| `ppo_epochs=1`       |         1.00         |        1.30        |     0.17     |       0.00       | 1m 48s |
| `kl_coef=0.01`       |         1.00         |        1.30        |     0.21     |       0.00       | 2m 27s |
| `kl_coef=0.20`       |         1.00         |        1.24        |     0.32     |       0.00       | 2m 47s |
| `clip_eps=0.10`      |         1.00         |        1.09        |     0.31     |       0.04       | 2m 25s |
| `grad_accum_steps=1` |         1.00         |        1.30        |     0.19     |       0.05       | 3m 37s |

The clearest way I found to make learning worse was to tighten the clipping too much. With `clip_eps=0.10`, the run still got the answers right, but the reward dropped to `1.09` and the clip fraction jumped to `0.04`, which to me is a good sign that the updates were running into the trust-region limit more often. In other words, the policy was trying to move, but the clipping threshold was tight enough that it started getting in the way.

The KL coefficient mattered too, but in a softer way. When I lowered `kl_coef` to `0.01`, it barely hurt anything on this easy task, while raising it to `0.20` lowered the reward to `1.24` and produced the largest KL number in the table. That sounds a little backwards at first, but on a tiny task like `format_copy`, the metrics are noisy enough that the penalty does not translate into a clean monotonic KL trend at the endpoint. What I do feel comfortable saying is that the larger KL setting was not helpful here.

Changing `ppo_epochs` from `2` to `1` mostly looked like an efficiency tradeoff. The run still reached perfect eval accuracy and top reward, but it finished faster, so on this task the second PPO pass was not obviously necessary. Setting `grad_accum_steps=1` made the updates more aggressive, which shows up in the larger clip fraction, but it did not buy any better final performance. Overall, the two knobs that seemed to matter most to me were clipping and how hard the run pushed against the trust region. On `format_copy`, the default settings were already close to a sweet spot, and the easiest way I found to make them worse was to make the update too conservative or too choppy.

## 5. Qualitative Behavior

What I found most informative qualitatively in `math_hard` was not just whether the model got the final number right, but how it chose to present that answer. By the end of training, GRPO was much better at consistently hitting the reward-relevant format: its final boxed-answer rate was `0.9141`, compared with `0.4102` for GR-REINFORCE. The sample tables also show something a little surprising. Even when the model gets full credit, it often does not follow the instruction to output only the boxed final answer. Instead, it still writes a long derivation and then ends with `\boxed{...}`.

- One GRPO example at step `499` came from the problem asking for `x^{\sqrt{2}} + y^{\sqrt{2}}` under the constraint `(\log_2 x)^4 + (\log_3 y)^4 + 8 = 8(\log_2 x)(\log_3 y)`. The completion correctly set `a = \log_2 x` and `b = \log_3 y`, worked through the algebra, and ended with `\boxed{13}`. What stood out to me was that this was still a long response, roughly `375` to `455` generated tokens in the logged GRPO samples, even though the system prompt explicitly says to output only the boxed answer. So GRPO clearly learned the important formatting target, but it did not really learn to be terse.

- One REINFORCE example at step `199` was even more surprising. On the centroid-area problem where the correct answer is `2`, the model also ended with the correct `\boxed{2}`, but one sampled completion included a Python-style scratch block and an `output 2.0` line before giving the final answer. To me, that is a nice illustration of what the reward is actually shaping. The model is being pushed hardest toward outcome correctness and the presence of a boxed answer, not toward clean obedience to the instruction to avoid extra prose. In other words, the reward got the model to care about the part that affects scoring, but not necessarily about elegance.
