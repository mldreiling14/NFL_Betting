# How the NFL Win Probability Model Works

A quick, plain-English rundown of what goes into predicting each game — no code, no math, just the "what" and "why."

---

## The basic idea

Before every NFL game, the model looks at a bunch of information about both teams — how they've been playing lately, who their key players are, their history, even the weather — and turns all of that into a percentage: "Team A has a 65% chance of winning."

It's not magic and it's not a sure thing. It's built the same way a smart, obsessive football fan would think about a game, just done with real numbers instead of gut feeling.

**How good is it?** On actual games it's never seen before, it picks the right winner about **68% of the time** — close to what Vegas sportsbooks manage (about 69%). It doesn't beat Vegas, but it's not far off either, which is a genuinely solid result for a side project.

---

## What it actually looks at

**How each team has been playing lately**
Not just their overall record for the season, but specifically their last 5 games — are they hot right now, or in a slump? Includes both wins/losses and how big the margins have been.

**A "power ranking" for every team (Elo)**
Similar to a chess rating. Every team gets a number that goes up when they win (especially against a good team) and down when they lose (especially to a weak one). This turns out to be the single most useful piece of information in the whole model — beating a great team means a lot more than beating a bad one, and this number captures that automatically.

**The starting quarterback's recent play**
QB performance is tracked individually, following that specific player — not just "how has this team done," but "how has *this player* been throwing the ball lately," including if they get traded or benched.

**Running back production**
How well a team's running backs have been performing recently, weighted by how much they're actually playing (so a backup's one big run doesn't count as much as a starter's steady output).

**Wide receivers and tight ends**
Same idea as running backs — how the team's pass-catchers have been performing lately, weighted by playing time.

**Injuries**
Whether the starting QB, top running back, or top receivers are banged up heading into the game (listed as questionable, doubtful, or out).

**Defense**
Rather than just counting sacks or interceptions (which can be a bit random game to game), this measures how many yards and "point value" a team's defense has actually been giving up to their opponents lately — a steadier, more meaningful signal.

**The offensive line**
How often the quarterback is getting sacked or pressured — a sign of whether the O-line is protecting well.

**Receiver vs. cornerback matchups**
Compares a team's top receiver (size, and who they're going up against) with the opposing team's top cornerback, including how well that cornerback has actually been covering people lately. Worth noting: we don't actually know who's covering whom on any given play — this estimates "best receiver vs. their most likely defender," which is a reasonable guess, not a certainty.

**Coaching history**
How these two specific head coaches have done against each other in the past, across their whole careers — some coaches just have a coach's number.

**Rest**
Whether one team has had more days off than the other coming into the game (like coming off a bye week).

**Home field**
Built in throughout — home teams have a real, measurable advantage, and the model accounts for it.

---

## What's *not* included (yet)

- **Weather** — built and tested, but it turned out not to reliably improve the predictions overall, even though there's a real pattern where warm-climate teams (like Miami) do somewhat worse when they travel into extreme cold. It's there in the data, just not currently part of the final number.
- **Live, in-game predictions** — this version only predicts *before* the game starts. A "win probability updating as the game happens" version (like you see on TV broadcasts) would be a different, future project.

---

## An honest note

This isn't a betting system, and it hasn't been shown to beat Vegas. When the model and Vegas disagree, Vegas is usually the one that's right. Think of it more as a well-reasoned, data-backed opinion — genuinely more informed than a coin flip, built carefully and tested honestly, but not a guarantee about what's going to happen on Sunday.
