import pandas as pd
import numpy as np
import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D projection

LEVEL_LABELS = ["similar", "neutral", "different"]
LEVEL_MAP = {
    "similar": 0,
    "neutral": 1,
    "different": 2,
}

def krippendorff_alpha_nominal(df: pd.DataFrame) -> float:
    """
    Compute Krippendorff's alpha (nominal) for a DataFrame with columns:
    'A', 'B', 'question', 'preferred', 'source_file'.

    Each unique (A, B, question) is an item. 'preferred' is 0/1.
    """
    # Drop rows without labels
    df = df.dropna(subset=['preferred']).copy()

    # Ensure one rating per (item, rater); if duplicates exist, keep first
    df = df.drop_duplicates(subset=['A', 'B', 'question', 'source_file'])

    # Build an item_id from A, B, question
    df['item_id'] = df[['A', 'B', 'question']].astype(str).agg('||'.join, axis=1)

    # Count how many times each category (0 or 1) was chosen per item
    counts = df.groupby(['item_id', 'preferred']).size().unstack(fill_value=0)

    # Number of ratings per item
    n_j = counts.sum(axis=1)

    # Keep only items with at least 2 ratings (needed for agreement)
    counts = counts[n_j >= 2]
    n_j = counts.sum(axis=1)

    if len(n_j) == 0:
        return np.nan  # not enough overlap to compute alpha

    # --------- Observed disagreement Do ---------
    # For each item j:
    #   D_o(j) numerator = sum_c n_cj (n_j - n_cj) = n_j^2 - sum_c n_cj^2
    num_Do = ((n_j ** 2 - (counts ** 2).sum(axis=1))).sum()
    den_Do = (n_j * (n_j - 1)).sum()
    Do = num_Do / den_Do

    # --------- Expected disagreement De ---------
    # n_c = total number of times category c was used across all items
    n_c = counts.sum(axis=0)
    N = n_c.sum()
    De = (N**2 - (n_c**2).sum()) / (N * (N - 1))

    alpha = 1 - Do / De if De != 0 else np.nan
    return float(alpha)

def krippendorff_alpha_by_question(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Krippendorff's alpha per unique 'question'.
    Returns a DataFrame with columns: question, alpha, n_items.
    """
    rows = []
    for q, df_q in df_all.groupby('question'):
        # Just reuse the function above on the subset
        alpha_q = krippendorff_alpha_nominal(df_q)
        n_items = df_q[['A', 'B', 'question']].drop_duplicates().shape[0]
        rows.append({'question': q, 'alpha': alpha_q, 'n_items': n_items})
    return pd.DataFrame(rows).sort_values('alpha', ascending=False)

def parse_design_levels(design_name: str):
    """
    Parse a design filename like:
        visually-different_semantically-neutral_compositionally-similar.png

    into numeric levels (vis, sem, comp) using LEVEL_MAP.
    """
    base = os.path.basename(design_name)
    stem, _ = os.path.splitext(base)  # drop .png

    # Expect something like "visually-X_semantically-Y_compositionally-Z"
    parts = stem.split("_")
    levels = {}

    for part in parts:
        if part.startswith("visually-"):
            levels["visually"] = part.split("-", 1)[1]
        elif part.startswith("semantically-"):
            levels["semantically"] = part.split("-", 1)[1]
        elif part.startswith("compositionally-"):
            levels["compositionally"] = part.split("-", 1)[1]

    # Basic safety check
    needed = {"visually", "semantically", "compositionally"}
    if not needed.issubset(levels.keys()):
        raise ValueError(f"Could not parse all levels from design name: {design_name}")

    try:
        vis = LEVEL_MAP[levels["visually"]]
        sem = LEVEL_MAP[levels["semantically"]]
        comp = LEVEL_MAP[levels["compositionally"]]
    except KeyError as e:
        raise ValueError(
            f"Unknown level {e} in design name: {design_name}. "
            f"Expected one of {list(LEVEL_MAP.keys())}."
        ) from e

    return vis, sem, comp

def fit_bradley_terry_from_df(df: pd.DataFrame,
                              question_filter: str | None = None,
                              lr: float = 0.05,
                              epochs: int = 1000) -> pd.DataFrame:
    """
    Fit a Bradley–Terry model on pairwise preference data already
    loaded in a DataFrame.

    df must have columns:
        - A, B: design identifiers (e.g., image paths)
        - question: which question was asked for this pair
        - preferred: 0 if A was preferred, 1 if B was preferred

    question_filter:
        - If not None, only use rows where df['question'] == question_filter.
        - If None, use all rows.

    Returns:
        DataFrame with columns: ['design', 'score', 'rank'] sorted by score (desc).
    """
    df = df.copy()

    # Optional: restrict to a specific question
    if question_filter is not None:
        df = df[df["question"] == question_filter].copy()
        if df.empty:
            raise ValueError(f"No rows found for question: {question_filter}")

    # Basic checks
    required_cols = {"A", "B", "preferred"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    # Map each design (path) to an integer id
    designs = sorted(set(df["A"]).union(df["B"]))
    design_to_idx = {d: i for i, d in enumerate(designs)}
    n_designs = len(designs)

    # Encode pairs as indices
    i_idx = df["A"].map(design_to_idx).to_numpy(dtype=np.int64)
    j_idx = df["B"].map(design_to_idx).to_numpy(dtype=np.int64)

    # preferred: 0 -> A preferred, 1 -> B preferred
    preferred_raw = df["preferred"].to_numpy()
    if not np.isin(preferred_raw, [0, 1]).all():
        raise ValueError("Column 'preferred' must contain only 0 (A) or 1 (B).")

    # We model P(A wins) = sigmoid(s_A - s_B),
    # so y = 1 if A preferred, 0 if B preferred
    y = 1.0 - preferred_raw.astype(np.float32)

    # Torch tensors
    i_idx_t = torch.from_numpy(i_idx)
    j_idx_t = torch.from_numpy(j_idx)
    y_t = torch.from_numpy(y)

    # Scores for each design (initialized to zero)
    scores = torch.zeros(n_designs, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([scores], lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()

        s_i = scores[i_idx_t]  # scores for A
        s_j = scores[j_idx_t]  # scores for B

        logits = s_i - s_j  # P(A wins) = sigmoid(s_i - s_j)
        loss = F.binary_cross_entropy_with_logits(logits, y_t)

        loss.backward()
        optimizer.step()

        # Optional: print some progress
        if (epoch + 1) % 100 == 0:
            with torch.no_grad():
                pred = (torch.sigmoid(logits) > 0.5).float()
                acc = (pred == y_t).float().mean().item()
            print(f"[Epoch {epoch+1:4d}] loss={loss.item():.4f}, train acc={acc:.3f}")

    # Extract final scores
    with torch.no_grad():
        final_scores = scores.detach().cpu().numpy()

    # Normalize scores (optional, just for readability)
    final_scores = final_scores - final_scores.mean()

    # Build ranking dataframe
    rank_order = np.argsort(-final_scores)  # descending
    ranks = np.empty_like(rank_order)
    ranks[rank_order] = np.arange(1, n_designs + 1)

    result_df = pd.DataFrame({
        "design": designs,
        "score": final_scores,
        "rank": ranks,
    }).sort_values("rank")

    return result_df

def fit_bradley_terry(csv_paths,
                      question_filter: str | None = None,
                      lr: float = 0.05,
                      epochs: int = 1000) -> pd.DataFrame:
    """
    Wrapper around fit_bradley_terry_from_df that accepts:
      - a single CSV path (str)
      - a list/tuple of CSV paths
      - a glob pattern (str, e.g. 'result/user_rating_*.csv')

    All CSVs must share the same schema (A, B, question, preferred).
    The rows are simply concatenated: BT will treat repeated pairs
    as repeated observations.
    """
    # Normalize input into a list of paths
    if isinstance(csv_paths, str):
        # If it's a string, we treat it as a glob pattern
        matched = glob.glob(csv_paths)
        if len(matched) == 0:
            # maybe it's a single explicit file path that doesn't match glob patterns
            if os.path.exists(csv_paths):
                paths = [csv_paths]
            else:
                raise FileNotFoundError(f"No files match: {csv_paths}")
        else:
            paths = matched
    else:
        # Assume it's an iterable of explicit paths
        paths = list(csv_paths)

    if not paths:
        raise ValueError("No CSV files provided or matched.")

    # Load and concat all CSVs
    df_list = []
    for p in paths:
        df_i = pd.read_csv(p)
        # Optional: keep track of which file each row came from
        df_i["source_file"] = os.path.basename(p)
        df_list.append(df_i)

    df_all = pd.concat(df_list, ignore_index=True)

    print(f"Loaded {len(df_all)} rows from {len(paths)} file(s).")

    # Overall inter-rater reliability across all items/questions
    alpha_overall = krippendorff_alpha_nominal(df_all)
    print(f"Overall Krippendorff's alpha (nominal): {alpha_overall:.3f}")

    # Optional: per-question reliability
    alpha_per_question = krippendorff_alpha_by_question(df_all)
    print("Krippendorff's alpha by question: ", alpha_per_question)

    # Fit BT on the combined DataFrame
    return fit_bradley_terry_from_df(
        df_all,
        question_filter=question_filter,
        lr=lr,
        epochs=epochs,
    )


def add_3d_coords(result_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add columns 'vis', 'sem', 'comp' (0/1/2) to result_df
    based on the 'design' filename.
    """
    coords = result_df["design"].apply(parse_design_levels)
    result_df[["vis", "sem", "comp"]] = pd.DataFrame(
        coords.tolist(), index=result_df.index
    )
    return result_df

def plot_design_space_3d(result_df: pd.DataFrame, title: str = ""):
    """
    3D scatter of designs in (vis, sem, comp) space:
      - x: visual level
      - y: semantic level
      - z: compositional level
      - color: Bradley–Terry score
      - star marker: best design
    """
    if not {"vis", "sem", "comp", "score"}.issubset(result_df.columns):
        raise ValueError("result_df must have columns: vis, sem, comp, score")

    # Find best design (max score)
    best_idx = result_df["score"].idxmax()
    best_row = result_df.loc[best_idx]

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    # All points
    sc = ax.scatter(
        result_df["vis"],
        result_df["sem"],
        result_df["comp"],
        c=result_df["score"],
        cmap="viridis",
        s=40,
        alpha=0.8,
    )

    # Highlight best
    ax.scatter(
        [best_row["vis"]],
        [best_row["sem"]],
        [best_row["comp"]],
        s=200,
        c="red",
        marker="*",
        edgecolor="k",
        linewidth=1.5,
        label="Best design",
    )

    # Axes labels and tick labels
    ax.set_xlabel("Visual similarity", labelpad=10)
    ax.set_ylabel("Semantic similarity", labelpad=10)
    ax.set_zlabel("Compositional similarity", labelpad=10)

    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_zticks([0, 1, 2])

    ax.set_xticklabels(LEVEL_LABELS)
    ax.set_yticklabels(LEVEL_LABELS)
    ax.set_zticklabels(LEVEL_LABELS)

    if title:
        ax.set_title(title, pad=20)

    cbar = plt.colorbar(sc, ax=ax, shrink=0.75)
    cbar.set_label("Bradley–Terry score")

    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()

    print("Best design:")
    print("  filename :", best_row['design'])
    print("  visual   :", LEVEL_LABELS[int(best_row['vis'])])
    print("  semantic :", LEVEL_LABELS[int(best_row['sem'])])
    print("  compos.  :", LEVEL_LABELS[int(best_row['comp'])])
    print("  score    :", best_row['score'])


def plot_design_space_3d_bubbles(result_df, title: str = ""):
    """
    3D scatter of designs:
      - Position: (vis, sem, comp)
      - Color: Bradley–Terry score
      - Size: score magnitude (bigger = more preferred)
    """

    if not {"vis", "sem", "comp", "score"}.issubset(result_df.columns):
        raise ValueError("result_df must have columns: vis, sem, comp, score")

    scores = result_df["score"].to_numpy()

    # Normalize scores to [0, 1] for sizing
    s_min, s_max = scores.min(), scores.max()
    if s_max > s_min:
        scores_norm = (scores - s_min) / (s_max - s_min)
    else:
        scores_norm = np.ones_like(scores) * 0.5  # fallback if all equal

    # Map normalized score to marker size (tweak min/max as you like)
    size_min, size_max = 50, 400
    sizes = size_min + scores_norm * (size_max - size_min)

    # Get best design
    best_idx = result_df["score"].idxmax()
    best_row = result_df.loc[best_idx]

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(
        result_df["vis"],
        result_df["sem"],
        result_df["comp"],
        s=sizes,
        c=scores,
        cmap="viridis",
        alpha=0.85,
    )

    # Optionally outline the best point
    ax.scatter(
        [best_row["vis"]],
        [best_row["sem"]],
        [best_row["comp"]],
        s=450,
        facecolors="none",
        edgecolors="red",
        linewidths=2.0,
        label="Best design",
    )

    # Axes labels and ticks
    ax.set_xlabel("Visual similarity", labelpad=10)
    ax.set_ylabel("Semantic similarity", labelpad=10)
    ax.set_zlabel("Compositional similarity", labelpad=10)

    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_zticks([0, 1, 2])

    ax.set_xticklabels(LEVEL_LABELS)
    ax.set_yticklabels(LEVEL_LABELS)
    ax.set_zticklabels(LEVEL_LABELS)

    if title:
        ax.set_title(title, pad=20)

    cbar = plt.colorbar(sc, ax=ax, shrink=0.75)
    cbar.set_label("Bradley–Terry score")

    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()

    print("Best design:")
    print("  filename :", best_row['design'])
    print("  visual   :", LEVEL_LABELS[int(best_row['vis'])])
    print("  semantic :", LEVEL_LABELS[int(best_row['sem'])])
    print("  compos.  :", LEVEL_LABELS[int(best_row['comp'])])
    print("  score    :", best_row['score'])


if __name__ == "__main__":
    result_df = fit_bradley_terry(
        [
            "result/user_rating_daehwa2.csv",
            "result/user_rating_Jack.csv",
            "result/user_rating_Angie.csv",
        ],
        # question_filter="Which drawing is more creative?",
    )

    result_df = add_3d_coords(result_df)

    plot_design_space_3d_bubbles(
        result_df,
        title="Preference landscape (bubble size = score)"
    )
    # plot_design_space_3d(
    #     result_df,
    #     title="Preference landscape: 'Which drawing is more creative?'"
    # )
