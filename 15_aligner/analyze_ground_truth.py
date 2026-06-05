import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from collections import Counter, defaultdict

if len(sys.argv) not in (3, 4):
    print(f"Usage: {sys.argv[0]} <trials.json> <manifest.json> [output_dir]")
    sys.exit(1)

data_path     = Path(sys.argv[1]).resolve()
manifest_path = Path(sys.argv[2]).resolve()
out_dir       = Path(sys.argv[3]).resolve() if len(sys.argv) == 4 \
                else data_path.parent / "ground_truth_stats"
out_dir.mkdir(parents=True, exist_ok=True)

with open(data_path) as f:
    trials = json.load(f)
with open(manifest_path) as f:
    manifest = json.load(f)

participant_name = trials[0]["participant"] if trials else "unknown"
participant_id   = manifest[0]["participant_id"] if manifest else "unknown"

# manifest: trial_index (str) -> n_windows (video phase)
weight_by_index = {
    int(e["trial_index"]): e["phases"]["video"]["n_windows"]
    for e in manifest
}

# ── Filter ────────────────────────────────────────────────────────────────────
# Skip: error non-empty  OR  emotion_tag_reference == "Trial"
valid = [
    t for t in trials
    if not t.get("error")
    and t.get("ground_truth", {}).get("emotion_tag_reference") != "Trial"
]

skipped_error     = sum(1 for t in trials if t.get("error"))
skipped_trial_ref = sum(
    1 for t in trials
    if not t.get("error")
    and t.get("ground_truth", {}).get("emotion_tag_reference") == "Trial"
)

valence = np.array([t["ground_truth"]["valence"] for t in valid])
arousal = np.array([t["ground_truth"]["arousal"] for t in valid])
weights = np.array([weight_by_index.get(t["index"] + 1, 1) for t in valid], dtype=float)

# ── Stats helpers ─────────────────────────────────────────────────────────────
def weighted_stats(arr, w):
    wm  = np.average(arr, weights=w)
    wv  = np.average((arr - wm) ** 2, weights=w)
    return float(wm), float(np.sqrt(wv))

def w_dist_dict(keys, w):
    d = defaultdict(float)
    for k, wt in zip(keys, w):
        d[k] += wt
    return d

def build_num_stats(arr, w):
    wm, ws = weighted_stats(arr, w)
    dist = Counter(arr.tolist())
    wd   = w_dist_dict(arr.tolist(), w.tolist())
    return {
        "count":                  int(len(arr)),
        "mean":                   round(float(arr.mean()), 4),
        "std":                    round(float(arr.std()),  4),
        "min":                    float(arr.min()),
        "max":                    float(arr.max()),
        "median":                 float(np.median(arr)),
        "distribution":           {str(int(k)): int(v) for k, v in sorted(dist.items())},
        "weighted_mean":          round(wm, 4),
        "weighted_std":           round(ws, 4),
        "weighted_distribution":  {str(int(k)): int(v) for k, v in sorted(wd.items())},
    }

# ── Emotion tag analysis ──────────────────────────────────────────────────────
def dominant_tag(emotion_tag: dict):
    """
    Returns (tag, reason_discarded).
    Tie-break: if neutral ties with exactly one non-neutral → pick non-neutral.
    Two+ non-neutral tied → discard ("tie_non_neutral").
    """
    if not emotion_tag:
        return None, "empty"
    max_val = max(emotion_tag.values())
    if max_val == 0:
        return None, "all_zero"
    tied = [k for k, v in emotion_tag.items() if v == max_val]
    if len(tied) == 1:
        return tied[0], None
    non_neutral = [k for k in tied if k != "neutral"]
    if len(non_neutral) == 1:
        return non_neutral[0], None
    return None, "tie_non_neutral"

tag_records   = []
skipped_tie   = 0

for t, w in zip(valid, weights):
    gt  = t["ground_truth"]
    tag, reason = dominant_tag(gt["emotion_tag"])
    if tag is None:
        skipped_tie += 1
        continue
    tag_records.append({
        "tag":       tag,
        "weight":    float(w),
        "reference": gt.get("emotion_tag_reference", ""),
    })

tag_names = [r["tag"]       for r in tag_records]
tag_w     = np.array([r["weight"]    for r in tag_records])
tag_refs  = [r["reference"] for r in tag_records]

tag_dist  = Counter(tag_names)
tag_wdist = w_dist_dict(tag_names, tag_w.tolist())

# Consistency per reference (excluding Trial, already filtered)
consistency = {}
for ref in sorted(set(tag_refs)):
    idxs   = [i for i, r in enumerate(tag_refs) if r == ref]
    total  = len(idxs)
    match  = sum(1 for i in idxs if tag_names[i].lower() == ref.lower())
    consistency[ref] = {
        "total": total,
        "match": match,
        "rate":  round(match / total, 4) if total else 0.0,
    }

# ── Build JSON result ─────────────────────────────────────────────────────────
result = {
    "participant_id":   participant_id,
    "participant_name": participant_name,
    "input": {
        "trials":   str(data_path),
        "manifest": str(manifest_path),
    },
    "filter": {
        "total":             len(trials),
        "skipped_error":     skipped_error,
        "skipped_trial_ref": skipped_trial_ref,
        "valid":             len(valid),
    },
    "valence": build_num_stats(valence, weights),
    "arousal": build_num_stats(arousal, weights),
    "emotion_tag": {
        "skipped_tie":              skipped_tie,
        "valid_for_tag":            len(tag_records),
        "distribution":             dict(sorted(tag_dist.items())),
        "weighted_distribution":    {k: int(tag_wdist[k]) for k in sorted(tag_wdist)},
        "consistency_by_reference": consistency,
    },
}

json_path = out_dir / "stats.json"
with open(json_path, "w") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)
print(f"Stats saved → {json_path}")

# ── Console summary ───────────────────────────────────────────────────────────
print(f"\nParticipant : {participant_name} (id={participant_id})")
print(f"Total       : {len(trials)}  |  skipped error={skipped_error}  "
      f"trial-ref={skipped_trial_ref}  |  valid={len(valid)}")
for label, d in [("Valence", result["valence"]), ("Arousal", result["arousal"])]:
    print(f"\n--- {label} ---")
    for k, v in d.items():
        print(f"  {k}: {v}")
et = result["emotion_tag"]
print(f"\n--- Emotion Tag ---")
print(f"  skipped_tie          : {et['skipped_tie']}")
print(f"  valid_for_tag        : {et['valid_for_tag']}")
print(f"  distribution         : {et['distribution']}")
print(f"  weighted_distribution: {et['weighted_distribution']}")
print(f"  consistency_by_reference:")
for ref, c in et["consistency_by_reference"].items():
    print(f"    {ref}: {c}")

# ── Shared style helpers ──────────────────────────────────────────────────────
COLORS = {
    "valence":   "#4C72B0", "arousal":    "#DD8452",
    "w_valence": "#7eb0d4", "w_arousal":  "#f5b97f",
}
TAG_PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B2", "#937860", "#DA8BC3",
]

def int_bins(arr):
    lo, hi = int(arr.min()), int(arr.max())
    return np.arange(lo - 0.5, hi + 1.5, 1.0)

def arr_wdist(arr, w):
    d = defaultdict(float)
    for v, wt in zip(arr.tolist(), w.tolist()):
        d[v] += wt
    keys = sorted(d)
    return np.array(keys), np.array([d[k] for k in keys])

title_base = (f"participant: {participant_name} (id={participant_id})  |  "
              f"valid n={len(valid)}, skipped error={skipped_error}, "
              f"trial-ref={skipped_trial_ref}")

# ════════════════════════════════════════════════════════════════════════════
# Plot 1 — Valence & Arousal
# ════════════════════════════════════════════════════════════════════════════
wv_mean, wv_std = weighted_stats(valence, weights)
wa_mean, wa_std = weighted_stats(arousal, weights)

fig1 = plt.figure(figsize=(14, 15))
fig1.suptitle(f"Valence & Arousal — {title_base}", fontsize=11, fontweight="bold")
gs1 = gridspec.GridSpec(3, 3, figure=fig1, hspace=0.50, wspace=0.35)

# 1. Valence histogram
ax = fig1.add_subplot(gs1[0, 0])
ax.hist(valence, bins=int_bins(valence), color=COLORS["valence"], edgecolor="white", rwidth=0.85)
ax.set_title("Valence — histogram")
ax.set_xlabel("Valence score"); ax.set_ylabel("Count (trials)")
ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

# 2. Arousal histogram
ax = fig1.add_subplot(gs1[0, 1])
ax.hist(arousal, bins=int_bins(arousal), color=COLORS["arousal"], edgecolor="white", rwidth=0.85)
ax.set_title("Arousal — histogram")
ax.set_xlabel("Arousal score"); ax.set_ylabel("Count (trials)")
ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

# 3. Scatter
ax = fig1.add_subplot(gs1[0, 2])
jitter = np.random.default_rng(0).uniform(-0.08, 0.08, size=(len(valid), 2))
ax.scatter(valence + jitter[:, 0], arousal + jitter[:, 1],
           alpha=0.6, s=40, color="#2ca02c", edgecolors="none")
ax.set_title("Valence vs Arousal")
ax.set_xlabel("Valence"); ax.set_ylabel("Arousal")
ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

# 4. Valence boxplot
ax = fig1.add_subplot(gs1[1, 0])
ax.boxplot(valence, vert=True, patch_artist=True,
           boxprops=dict(facecolor=COLORS["valence"], alpha=0.6))
ax.set_title("Valence — box plot"); ax.set_ylabel("Score")
ax.set_xticks([1]); ax.set_xticklabels(["Valence"])
ax.text(1.35, valence.mean(), f"μ={valence.mean():.2f}\nσ={valence.std():.2f}",
        va="center", fontsize=9)

# 5. Arousal boxplot
ax = fig1.add_subplot(gs1[1, 1])
ax.boxplot(arousal, vert=True, patch_artist=True,
           boxprops=dict(facecolor=COLORS["arousal"], alpha=0.6))
ax.set_title("Arousal — box plot"); ax.set_ylabel("Score")
ax.set_xticks([1]); ax.set_xticklabels(["Arousal"])
ax.text(1.35, arousal.mean(), f"μ={arousal.mean():.2f}\nσ={arousal.std():.2f}",
        va="center", fontsize=9)

# 6. Summary table
ax = fig1.add_subplot(gs1[1, 2]); ax.axis("off")
rows = [
    ["", "Valence", "Arousal"],
    ["n",        str(len(valence)),          str(len(arousal))],
    ["mean",     f"{valence.mean():.2f}",    f"{arousal.mean():.2f}"],
    ["std",      f"{valence.std():.2f}",     f"{arousal.std():.2f}"],
    ["min",      f"{valence.min():.1f}",     f"{arousal.min():.1f}"],
    ["max",      f"{valence.max():.1f}",     f"{arousal.max():.1f}"],
    ["median",   f"{np.median(valence):.2f}",f"{np.median(arousal):.2f}"],
    ["w.mean",   f"{wv_mean:.2f}",           f"{wa_mean:.2f}"],
    ["w.std",    f"{wv_std:.2f}",            f"{wa_std:.2f}"],
    ["Σ windows",f"{int(weights.sum())}",    f"{int(weights.sum())}"],
]
tbl = ax.table(cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.15, 1.35)
ax.set_title("Summary statistics")

# 7. Weighted valence bar
ax = fig1.add_subplot(gs1[2, 0])
vkeys, vcounts = arr_wdist(valence, weights)
bars = ax.bar(vkeys, vcounts, color=COLORS["w_valence"], edgecolor="white", width=0.7)
ax.axvline(wv_mean, color=COLORS["valence"], linewidth=1.8, linestyle="--",
           label=f"w.μ={wv_mean:.2f}")
ax.set_title("Valence — weighted (n_windows)")
ax.set_xlabel("Valence score"); ax.set_ylabel("Total windows")
ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True)); ax.legend(fontsize=9)
for bar, cnt in zip(bars, vcounts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            str(int(cnt)), ha="center", va="bottom", fontsize=8)

# 8. Weighted arousal bar
ax = fig1.add_subplot(gs1[2, 1])
akeys, acounts = arr_wdist(arousal, weights)
bars = ax.bar(akeys, acounts, color=COLORS["w_arousal"], edgecolor="white", width=0.7)
ax.axvline(wa_mean, color=COLORS["arousal"], linewidth=1.8, linestyle="--",
           label=f"w.μ={wa_mean:.2f}")
ax.set_title("Arousal — weighted (n_windows)")
ax.set_xlabel("Arousal score"); ax.set_ylabel("Total windows")
ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True)); ax.legend(fontsize=9)
for bar, cnt in zip(bars, acounts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            str(int(cnt)), ha="center", va="bottom", fontsize=8)

# 9. n_windows per trial
ax = fig1.add_subplot(gs1[2, 2])
sorted_idx = np.argsort(weights)
ax.barh(np.arange(len(weights)), weights[sorted_idx], color="#9ecae1", edgecolor="none")
ax.set_title("n_windows per valid trial")
ax.set_xlabel("n_windows (video phase)"); ax.set_ylabel("Trial rank (by weight)")
ax.text(0.98, 0.02, f"Σ={int(weights.sum())}\nμ={weights.mean():.1f}",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=9)

p1 = out_dir / "valence_arousal_stats.png"
fig1.savefig(p1, dpi=150, bbox_inches="tight")
plt.close(fig1)
print(f"Plot 1 saved → {p1}")

# ════════════════════════════════════════════════════════════════════════════
# Plot 2 — Emotion Tag
# ════════════════════════════════════════════════════════════════════════════
all_tags  = sorted(tag_dist.keys())
tag_color = {t: TAG_PALETTE[i % len(TAG_PALETTE)] for i, t in enumerate(sorted(set(all_tags)))}

fig2 = plt.figure(figsize=(14, 10))
fig2.suptitle(
    f"Emotion Tag — {title_base}\n"
    f"(tag valid={len(tag_records)}, skipped tie={skipped_tie})",
    fontsize=11, fontweight="bold"
)
gs2 = gridspec.GridSpec(2, 3, figure=fig2, hspace=0.55, wspace=0.38)

# 1. Tag count distribution
ax = fig2.add_subplot(gs2[0, 0])
tags_sorted = sorted(tag_dist, key=lambda k: -tag_dist[k])
counts      = [tag_dist[t] for t in tags_sorted]
colors_bar  = [tag_color[t] for t in tags_sorted]
bars = ax.bar(tags_sorted, counts, color=colors_bar, edgecolor="white")
ax.set_title("Emotion tag — count")
ax.set_xlabel("Tag"); ax.set_ylabel("Count (trials)")
ax.tick_params(axis="x", rotation=30)
for bar, cnt in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            str(cnt), ha="center", va="bottom", fontsize=9)

# 2. Tag weighted distribution
ax = fig2.add_subplot(gs2[0, 1])
tags_wsorted = sorted(tag_wdist, key=lambda k: -tag_wdist[k])
wcounts      = [int(tag_wdist[t]) for t in tags_wsorted]
colors_wbar  = [tag_color[t] for t in tags_wsorted]
bars = ax.bar(tags_wsorted, wcounts, color=colors_wbar, edgecolor="white")
ax.set_title("Emotion tag — weighted (n_windows)")
ax.set_xlabel("Tag"); ax.set_ylabel("Total windows")
ax.tick_params(axis="x", rotation=30)
for bar, cnt in zip(bars, wcounts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            str(cnt), ha="center", va="bottom", fontsize=9)

# 3. Pie chart of tag distribution
ax = fig2.add_subplot(gs2[0, 2])
pie_tags   = sorted(tag_dist, key=lambda k: -tag_dist[k])
pie_counts = [tag_dist[t] for t in pie_tags]
pie_colors = [tag_color[t] for t in pie_tags]
ax.pie(pie_counts, labels=pie_tags, colors=pie_colors,
       autopct="%1.0f%%", startangle=90, textprops={"fontsize": 9})
ax.set_title("Emotion tag — proportion")

# 4. Consistency per reference — match vs mismatch stacked bar
ax = fig2.add_subplot(gs2[1, 0:2])
refs       = sorted(consistency.keys())
totals     = [consistency[r]["total"] for r in refs]
matches    = [consistency[r]["match"] for r in refs]
mismatches = [t - m for t, m in zip(totals, matches)]
rates      = [consistency[r]["rate"] for r in refs]
x = np.arange(len(refs))
b1 = ax.bar(x, matches,    color="#55A868", label="match",    edgecolor="white")
b2 = ax.bar(x, mismatches, bottom=matches, color="#C44E52", label="mismatch", edgecolor="white")
ax.set_title("Consistency: dominant tag vs emotion_tag_reference")
ax.set_xlabel("emotion_tag_reference"); ax.set_ylabel("Count (trials)")
ax.set_xticks(x); ax.set_xticklabels(refs, rotation=30, ha="right")
ax.legend(fontsize=9)
for i, (tot, rate) in enumerate(zip(totals, rates)):
    ax.text(i, tot + 0.1, f"{rate*100:.0f}%", ha="center", va="bottom",
            fontsize=9, fontweight="bold")

# 5. Consistency table
ax = fig2.add_subplot(gs2[1, 2]); ax.axis("off")
tbl_rows = [["reference", "total", "match", "rate%"]]
for ref in refs:
    c = consistency[ref]
    tbl_rows.append([ref, str(c["total"]), str(c["match"]),
                     f"{c['rate']*100:.1f}%"])
tbl = ax.table(cellText=tbl_rows[1:], colLabels=tbl_rows[0],
               loc="center", cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.1, 1.6)
ax.set_title("Consistency table")

p2 = out_dir / "emotion_tag_stats.png"
fig2.savefig(p2, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"Plot 2 saved → {p2}")
