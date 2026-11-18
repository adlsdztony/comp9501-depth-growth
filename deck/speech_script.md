# Speech script — "Grow instead of search?"

Written for the deck `COMP9501_Growth_Results_v9(4).pptx` (17 slides).
Speaker 1: slides 1–5 · Speaker 2: slides 6–10 · Speaker 3: slides 11–17.
Roughly 4 minutes per speaker at a relaxed pace. Words in **bold** are worth stressing.

---

## Speaker 1 — slides 1 to 5

### Slide 1 · Title

Hi everyone. Our project is called **"Grow instead of search?"**

The question is a very practical one. When you train a reinforcement-learning policy, you have to decide how deep the network should be. Usually nobody knows the right answer in advance. So we asked: instead of training several networks and picking the best one, can we just **grow one network** while it trains?

### Slide 2 · Motivation

Here's the usual situation. Depth is a hyperparameter. If the network is too shallow, it can't learn the task well. If it's too deep, you might be wasting compute.

The standard answer is **search**. You train one network per candidate depth — that's the left side of this picture — and you keep the one that does best on validation data. The problem is that the cost grows with the number of candidates. Four depths means four full training runs. And in RL, one run is already millions of simulator steps, so this gets expensive fast.

The alternative is on the right: **growth**. You start with a small network, and you add layers along the way. That way you only pay for **one** run.

So the question at the bottom is our main question: can one growing run match a searched depth, at a lower total cost?

### Slide 3 · Method

So how do we add a layer without breaking what the policy has already learned?

The trick is really simple. The top row is the network before growth. The hidden layer ends with a ReLU, so the hidden values *h* are always **zero or positive**.

In the bottom row, we insert a new layer right there — shown in orange. It's a square linear layer, initialised to the **identity matrix**, followed by another ReLU. Since *h* is already non-negative, ReLU of *h* is just *h*. So the new layer passes everything through unchanged, and the policy is **exactly the same** at the moment we insert it.

A couple of details: the old weights keep their optimiser state, the new layer starts fresh, and we don't touch the critic. We checked all **twenty** insertions in our experiments, and every single one changed the policy by exactly zero. So, no insertion shock.

### Slide 4 · Benchmark

To test this, we use **Meta-World MT10**. These pictures are actual frames from the benchmark. It's one simulated robot arm, and **one network** has to solve all ten tasks — reaching, pushing, pick-and-place, opening doors, drawers, windows, and so on.

The network gets the robot state plus a one-hot ID that says which task it's on. It outputs four numbers: how to move the arm and the gripper. Each episode is 500 steps, and it either succeeds or it doesn't.

Some tasks are easy, like reach. Others need careful contact with objects, like push, peg insertion and pick-and-place — and those turn out to be the interesting ones.

### Slide 5 · Algorithm

The learner is **multi-task PPO**. Very quickly: ten environments, one per task, collect a batch of experience. We compute advantages, then update both the actor and the critic for several epochs. PPO's clipped objective keeps each update small, so training stays stable.

We also use **PopArt** in the critic. Different tasks give very different reward scales, and PopArt normalises them so one task doesn't dominate.

The key point is at the bottom: **every run uses the same learner and the same budget**. The only thing we change is the depth of the actor — and, for the growth runs, when it grows.

Now [Speaker 2] will walk you through the setup and our first findings.

---

## Speaker 2 — slides 6 to 10

### Slide 6 · Setup

Thanks. So here's the protocol.

Every run gets **three million** environment steps. The actor has width 400 and depth from one to four. The critic is fixed.

One thing we were careful about is evaluation. Each task has 50 versions with different goal and object positions — that's the bar on the right. We train on 30, use 10 for **validation**, so for picking models, and keep 10 completely **held out** for testing. Every number you'll see from now on is on those held-out goals.

We ran five seeds per configuration, so 43 training runs in total.

### Slide 7 · Baselines

Next, what does growth compete against?

First, two **search** methods. **Grid search** trains all four depths fully and picks the best one on validation. That costs four runs, so 12 million steps. **Successive halving** is a cheaper search. It trains all four depths for a short time, drops the worse half, keeps going, and in the end only one depth gets the full budget. That costs 5.5 million steps. We simulate it on the learning curves we already logged, so it sees exactly the same runs.

Second, two **no-search** baselines: just using the default depth, which is 2, or picking a random depth. We also show the best depth in hindsight, but only as a reference. You can't actually get that without searching.

And third, our two **growth** schedules. Both start at depth 1 and grow to depth 3. **Grow early** adds layers at half a million and one million steps. **Grow late** adds them at one million and two million.

And how do we count cost? It's all the compute you spend **before** you have your final model — including every run a search throws away.

### Slide 8 · Finding 1

Our first finding is a bit of a side result. On the left are training goals, on the right held-out goals, for every configuration. And they're almost the same — the gap is **at most about four points**.

That's because the goal positions within a task are quite similar to each other. So the held-out test is fair, but it doesn't make the benchmark harder. What actually makes a difference is depth, which is the next slide.

### Slide 9 · Finding 2

Depth really matters here. Going from **one** layer to **four**, average success goes from **64%** to **88%**. So picking the wrong depth can cost you up to **24 points**.

The heat map shows where that comes from. The easy tasks, like reach or the windows, are solved at every depth. The gains come from the hard, contact-rich tasks: peg insertion goes from 0 to 76%, push from 22 to 88%, and drawer-open from 40 to 100%.

So choosing the depth is worth paying for. The question is how much you have to pay.

### Slide 10 · Growth dynamics

Here's what growth looks like during training. Each bold line is one seed of a growth run, and the dotted lines mark the moments we insert a layer. The thin lines are fixed depth 1 and depth 3, for comparison.

First thing: there's **no dip** after any insertion. That matches what we said about the identity layer.

Second: **timing matters**. On the left, grow early has two million steps at full depth, and it catches up with the deep networks. On the right, grow late only has one million steps left, and it ends up around depth-2 level.

So the lesson is: **when** you grow matters more than **whether** you grow.

Now [Speaker 3] will show how this compares with search.

---

## Speaker 3 — slides 11 to 17

### Slide 11 · Compute budget

Thanks. This slide shows the cost side.

The x-axis is the **total compute spent so far**, including every run a search method is training. The y-axis is how good the model is that each method would pick at that point.

Look at the purple curve: grow early reaches **84%** after about 0.18 petaFLOPs. At that same spend, successive halving — the green curve — is only around **60%**, and grid search is even lower. That's because search is paying for four networks at once, while growth is paying for one.

Search does catch up, but only after spending about **twice** as much compute.

### Slide 12 · Main result

Here's the final comparison, averaged over five seeds. Success is on the left, compute on the right.

Grid search gets the best result, **89%**, but it costs **0.71** petaFLOPs, because it trains four networks. Successive halving gets **85%** for **0.36**. Grow early gets **84%** for only **0.18** — so about the same as successive halving, at **half the compute**.

It's also better than just using the default depth or a random depth.

But to be honest, it's still about **5 points below** the best fixed depth. So one growing run is about as good as a cheap search, but not as good as a full search.

### Slide 13 · Compute

Why is growth so cheap? Because it spends the beginning of training in a **small** network.

We measure cost as learner FLOPs, which counts the forward and backward passes of the actor and critic. Each extra layer adds cost: depth 4 costs about **2.3 times** depth 1 per run.

Grow early costs **0.18** petaFLOPs. That's less than even a single run of depth 3 or depth 4, because for the first million steps it's training a smaller network. And FLOPs don't depend on which GPU we used, so the comparison is fair across machines.

### Slide 14 · Negative result

We also want to show where growth **doesn't** work. With a narrower actor, width 128, growth didn't help at all. Grow early got **71%** and grow late **66%**. Both are below even a random depth, which got 72%.

We only have two seeds here, so we see this as a warning rather than a conclusion. But it tells us growth is not a free lunch: whether it helps depends on the width and on the schedule.

### Slide 15 · Limitations

A few limitations to keep in mind.

On **scope**: it's one benchmark, one algorithm, and a three-million-step budget. Longer training could change the ranking.

On **method**: our growth follows a fixed schedule, not an adaptive rule. And successive halving is simulated from our logged curves.

On **measurement**: the held-out goals test new positions, not new tasks. And each evaluation has 100 episodes, so a single run's number has about plus or minus four points of noise.

So please read this as evidence **for growing early**, not as a finished method.

### Slide 16 · Takeaways

To wrap up.

First, identity insertion lets you **grow a policy safely**: zero change at all twenty insertions, and no drop afterwards.

Second, if you grow **early**, one run performs about as well as successive halving, at half the compute. It's still five points below the best depth, and it didn't help with a narrow network.

And for next steps, the big question is **deciding when to grow automatically** — for example, adding a layer when learning plateaus — and also trying deeper growth, more seeds, and the bigger MT50 benchmark.

### Slide 17 · Thank you

That's our project. Thank you for listening — we're happy to take questions.

---

## Quick answers for likely questions

**"Why not just train depth 4? It was best."** In our runs, yes — but you only know that after searching. The point is that the right depth is unknown in advance. At width 128, depth 4 was actually *worse* than depth 3.

**"Is identity insertion new?"** No. It's the same idea as Net2Net's deepening. What's different is the question: we use it to replace a depth search, and we compare it against search with the cost of the search included.

**"Isn't successive halving simulated?"** Yes. It uses the same logged learning curves a real run would see, but we didn't tune its schedule. We also tried 24 different schedules (backup slide), and none of them is as cheap as grow early at the same success.

**"Why FLOPs and not GPU time?"** Our networks are small, so GPU time was dominated by overhead and server load. It barely changed with depth. FLOPs measure the actual work and don't depend on the hardware.

**"Is 84% vs 85% significant?"** No. With five seeds we can't tell them apart. The claim is that they're comparable, not that growth is better.
