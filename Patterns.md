# Patterns Recognized

**One.** Knobs at top. All settings in one block. Change there, not below.

**Two.** Small hot code. Dot, add, relu run a lot. Keep them tiny. Let them inline.

**Three.** Outer loop splits. Rows do not touch. Spread rows over cores.

**Four.** Softmax takes away max. Big scores blow up. Take max first. Then exp. Then divide.

**Five.** Front and back are twins. Each forward step has a back step. Match them.

**Six.** Add paths let flow. Skip links carry blame back. Deep nets need them.

**Seven.** Clip big steps. Sum all blame. If too big, shrink. Keep under a cap.

**Eight.** Rate goes up, flat, down. Warm start. Hold. Cool at end.

**Nine.** State stays. Past blame and slow copies live in buffers. Do not reset.

**Ten.** Seed once. Same seed, same run. Same start weights.

**Eleven.** Read file by line. Track quote and slash. Simple beats smart.

**Twelve.** Skip dot and dot-dot. Folders lie about their name.

**Thirteen.** Bytes not words. Byte split never fails. Unknown text still works.

**Fourteen.** Count first, build next. Guess size, ask, then go.

**Fifteen.** Fail soft. Bad file? Skip. Bad thread? Fall back.

**Sixteen.** Log often. Print loss. Print guess. Print time.

**Seventeen.** One change per run. Then compare. Two changes hide the cause.

**Eighteen.** Ground truth first. Check by hand. Then trust the curve.

**Nineteen.** Guard zero. Empty list, flat row, small sum. Have a fallback.

**Twenty.** Copy before change. Sample, then pick. Do not edit the source.

**Twenty-one.** Sum then scale. Two passes beat one. Fast, then fix size.

**Twenty-two.** Test one weight. Nudge by a hair. Compare two sides. Small check, big save.

**Twenty-three.** Read the shape. M by K, K by N. Wrong shape, wrong math.

**Twenty-four.** Plain names win. `dW` not `delta_weight_buffer`. Short reads fast.
