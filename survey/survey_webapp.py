
# survey_webapp.py
#
# Minimal web port of your DearPyGui + VLC survey to a browser.
# - Framework: NiceGUI (pip install nicegui==2.*)
# - URL format: http://<server>:8080/<pname>  (e.g., /p1, /alice, /u001)
# - Data files:
#     - Base CSV: result/user_rating.csv           (the source template without _<pname>)
#     - Per-user CSV: result/user_rating_<pname>.csv
#     - Videos: dataset/<scene_id>/RGB.mp4           (served as static files)
#
# Run:
#   pip install nicegui pandas
#   python survey_webapp.py
#   (then open http://localhost:8080/p1)
#
# Optional production tip:
#   uvicorn survey_webapp:app --host 0.0.0.0 --port 8080
#
# Notes:
# - Concurrency: each participant should use a unique <pname>, which writes a separate CSV.
# - Sorting and column semantics mirror your original script (removes column index 1 from sorting order).
# - This uses HTML5 <video> (no VLC); videos must be browser-compatible (e.g., H.264 .mp4).
#
from __future__ import annotations

import os
import threading
from typing import List, Dict, Any
from datetime import datetime
import time

import pandas as pd
from nicegui import ui, app
from pathlib import Path
import numpy as np
from itertools import combinations
from collections import defaultdict
from typing import List, Tuple
import random

from questionnaire_list import questionnaire_lst

# ---- Config ----
RESULT_DIR        = 'result/'
DATASET_DIR       = 'data'
REQUIRED_COLUMNS  = ["A", "B", "question", "preferred"]

APP_DIR = Path(__file__).parent.resolve()
DATASET_ABS = (APP_DIR / DATASET_DIR).resolve()
app.add_static_files("/"+DATASET_DIR, str(DATASET_DIR))

MODE_TEXT = "text"
MODE_CI   = "correct_incorrect"
MODE_AB   = "ab"
# UI_ORDER  = [MODE_TEXT, MODE_CI, MODE_CI, MODE_AB]
UI_ORDER  = [MODE_AB]
SCENE_PREFIX = "Ground-truth scene description:"

BASE_DIR = Path(__file__).parent

# ---- Utilities to mirror original behavior ----
def save_to_csv(csv_path: str, df: pd.DataFrame) -> None:
    columns = df.columns.tolist()
    if len(columns) > 1:
        # match your "columns.pop(1)" before sorting
        columns.pop(1)
    df = df.sort_values(by=columns, ascending=True, ignore_index=True)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)

def make_balanced_pairs(
    designs: List[str],
    min_degree: int = 8,
    random_state: int = 0,
) -> List[Tuple[str, str]]:
    """
    Given a list of design identifiers (e.g., image paths),
    generate a set of unordered pairs (A,B) such that
    each design appears in at least `min_degree` pairs,
    as much as possible.

    Returns a list of (design_A, design_B).
    """
    rng = random.Random(random_state)
    n = len(designs)
    if n < 2:
        raise ValueError(f"Need at least 2 designs, got {n}")
    if min_degree > n - 1:
        raise ValueError(
            f"min_degree={min_degree} impossible with n={n} "
            f"(each design has at most n-1 neighbors)."
        )

    # Work with indices, then map back to designs at the end
    degrees = [0] * n
    pair_set = set()  # store pairs as (i, j) with i < j

    # Greedy algorithm:
    #   repeatedly go through all designs and, if its degree is too low,
    #   connect it to a random other design it hasn't been paired with yet.
    while True:
        progress = False

        for i in range(n):
            if degrees[i] >= min_degree:
                continue

            # Find available partners j for i
            candidates = []
            for j in range(n):
                if i == j:
                    continue
                a, b = (i, j) if i < j else (j, i)
                if (a, b) not in pair_set:
                    candidates.append((a, b))

            if not candidates:
                # No more unused partners for i
                continue

            # Choose one candidate pair at random
            a, b = rng.choice(candidates)
            pair_set.add((a, b))
            degrees[a] += 1
            degrees[b] += 1
            progress = True

        if not progress:
            # No new pairs could be added in this pass.
            # Either all designs reached min_degree, or we hit structural limits.
            break

        # Optional: stop early if everyone is at/above min_degree
        if all(d >= min_degree for d in degrees):
            break

    # Map indices back to design IDs
    pair_list = [(designs[i], designs[j]) for (i, j) in pair_set]

    # You can print some diagnostics:
    print(f"Generated {len(pair_list)} pairs for {n} designs.")
    print("Degree stats: min =", min(degrees), "max =", max(degrees),
          "avg =", sum(degrees) / n)

    return pair_list

def parent_prefix(rel_path: str, depth: int) -> str:
    parts = rel_path.split(os.sep)
    dir_parts = parts[:-1]          # drop filename
    key_parts = dir_parts[:depth]   # first N directories
    return os.path.join(*key_parts) if key_parts else ""

def create_template(
    csv_path: str,
    use_all_pairs: bool = True,
    min_degree: int = 8,
    random_state: int = 0,
) -> pd.DataFrame:
    """
    Treat each PNG file under DATASET_DIR as an atomic design.

    If use_all_pairs=True:
        - Use all unordered pairs of designs (full factorial).

    If use_all_pairs=False:
        - Use a sparse set of pairs chosen by make_balanced_pairs() so that
          each design appears in at least `min_degree` pairs (as much as possible).

    Then for each question in `questionnaire_lst`, replicate the pairs.

    The resulting DataFrame has columns: A, B, question.

    If csv_path exists, merge and deduplicate by unordered (A,B,question).
    """
    # 1. Collect all PNG designs as relative paths from DATASET_DIR
    designs = []
    for subdir, dirs, files in os.walk(DATASET_DIR):
        for file in files:
            if file.endswith(".png") and "turn_2_robot" in file:
                rel_dir = os.path.relpath(subdir, DATASET_DIR)
                if rel_dir == ".":
                    rel_path = file
                else:
                    rel_path = os.path.join(rel_dir, file)
                designs.append(rel_path)

    designs = sorted(set(designs))
    if len(designs) < 2:
        raise ValueError(f"Need at least 2 designs, found {len(designs)}")

    groups = defaultdict(list)
    for d in designs:
        key = parent_prefix(d, 1)
        groups[key].append(d)

    pair_list = []
    for key, group_designs in groups.items():
        if len(group_designs) < 2:
            continue  # nothing to pair inside this group
        # all unordered pairs *within* this group
        pair_list.extend(combinations(sorted(group_designs), 2))


    # # 2. Build pair list: full or sparse
    # if use_all_pairs:
    #     # Full factorial over designs
    #     pair_list = list(combinations(designs, 2))  # all unordered pairs
    #     print(f"Using ALL pairs: {len(pair_list)} pairs for {len(designs)} designs.")
    # else:
    #     # Sparse, coverage-guaranteed pairs
    #     pair_list = make_balanced_pairs(
    #         designs, min_degree=min_degree, random_state=random_state
    #     )
    #     print(f"Using SPARSE pairs: {len(pair_list)} pairs for {len(designs)} designs.")

    # 3. Expand by questions
    rows = []
    for q in questionnaire_lst:
        for a, b in pair_list:
            rows.append({"A": a, "B": b, "question": q})

    df_new = pd.DataFrame(rows, columns=["A", "B", "question"])

    # 4. Merge with existing CSV if it exists, dedup unordered pairs per question
    if os.path.exists(csv_path):
        df_old = pd.read_csv(csv_path)

        # Ensure needed columns exist
        for col in ["A", "B", "question"]:
            if col not in df_old.columns:
                df_old[col] = np.nan

        df_all = pd.concat([df_old, df_new], ignore_index=True)

        # Normalize (A,B) so unordered pairs are treated the same
        AB_sorted = np.sort(df_all[["A", "B"]].values, axis=1)
        df_all["A_norm"] = AB_sorted[:, 0]
        df_all["B_norm"] = AB_sorted[:, 1]

        df_template = (
            df_all
            .drop_duplicates(subset=["A_norm", "B_norm", "question"], keep="first")
            .drop(columns=["A_norm", "B_norm"])
        )
    else:
        df_template = df_new

    return df_template

def open_saved_file(csv_path: str, pname: str) -> pd.DataFrame:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    # # Load or create
    # if os.path.exists(csv_path):
    #     df = pd.read_csv(csv_path)
    # else:
        # copy base, append required columns
    df = create_template(csv_path, use_all_pairs=False)
    for c in REQUIRED_COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA
    save_to_csv(csv_path, df)

    return df

def get_imcomplete_row_indices(df: pd.DataFrame) -> List[int]:
    missing = df[REQUIRED_COLUMNS].isnull().any(axis=1)
    incomplete_rows = df[missing].copy()
    groups = [g for _, g in incomplete_rows.groupby(["A", "B"], sort=False)]
    rng = random.Random(int(time.time()))
    rng.shuffle(groups)
    shuffled = pd.concat(groups)
    return shuffled.index.to_list()

# ---- Per-participant state ----
class SessionState:
    def __init__(self, pname: str):
        self.pname = pname
        self.csv_path = f'{RESULT_DIR}/user_rating_{pname}.csv'
        self.df = open_saved_file(self.csv_path, pname)
        self.df_indices = get_imcomplete_row_indices(self.df)
        self.isswapped = False
        self.prev_a, prev_b = '', ''
        if not self.df_indices:
            self.df_indices = []  # no tasks left
            self.df_idx = None
        else:
            self.df_idx = self.df_indices.pop(0)
            if np.random.rand() < 0.5:
                self.isswapped = True
            else:
                self.isswapped = False
            self.prev_a, self.prev_b = self.current_pair()

        self.ui_idx = 0  # index into UI_ORDER
        self.lock = threading.Lock()  # file updates

    def current_pair(self):
        if self.df_idx is None:
            return None, None
        if not self.isswapped:
            a = self.df.iloc[self.df_idx]["A"]
            b = self.df.iloc[self.df_idx]["B"]
        else:
            b = self.df.iloc[self.df_idx]["A"]
            a = self.df.iloc[self.df_idx]["B"]
        return a, b

    def file_path_for_current(self) -> str | None:
        a, b = self.current_pair()
        if a is None or b is None:
            return None, None

        # URL for the browser
        return f'/{DATASET_DIR}/{a}', f'/{DATASET_DIR}/{b}'

    def prompt_for_current(self) -> str | None:
        a, _ = self.current_pair()
        return None if a is None else str(self.df[self.df_idx].iloc[2])

    def get_questionnaire(self) -> str | None:
        dat = self.df.iloc[self.df_idx]
        if dat is None:
            return None
        desc = str(dat.iloc[2])
        return desc

    def advance(self):
        if not self.df_indices:
            self.df_idx = None
            return
        self.ui_idx = (self.ui_idx + 1) % len(UI_ORDER)
        if self.ui_idx == 0:
            if not self.df_indices:
                self.df_idx = None
            else:
                self.df_idx = self.df_indices.pop(0)
                curr_a, curr_b = self.current_pair()
                if curr_a != self.prev_a or curr_b != self.prev_b:
                    if np.random.rand() < 0.5:
                        self.isswapped = True
                    else:
                        self.isswapped = False
                self.prev_a, self.prev_b = self.current_pair()

# ---- NiceGUI plumbing ----
# Serve dataset folder as /dataset so <video> can play files directly
app.add_static_files(f'/{DATASET_DIR}', DATASET_DIR)

# In-memory registry of sessions by participant name
sessions: Dict[str, SessionState] = {}

def get_session(pname: str) -> SessionState:
    if pname not in sessions:
        sessions[pname] = SessionState(pname)
    return sessions[pname]

def submit_event(state: SessionState, event: Dict[str, Any]) -> None:
    """Mirror your on_event() to update DF and save."""
    with state.lock:
        df = state.df
        idx = state.df_idx
        if idx is None:
            return

        pref = event['value']
        val = 0 if pref == 'A' else 1

        if state.isswapped:
            df.at[idx, "preferred"] = 1 - val
        else:
            df.at[idx, "preferred"] = val

        save_to_csv(state.csv_path, df)

        state.advance()

@ui.page('/{pname}')
def survey_page(pname: str):
    from datetime import datetime

    state = get_session(pname)

    # --- top-level layout elements (must NOT be nested) ---
    with ui.header().classes('p-3'):
        ui.label(f'Participant: {state.pname}').style('font-weight:600')
        ui.space()
        tasks_lbl = ui.label('')
        clock = ui.label(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    tasks_lbl.set_text(f'Tasks left: {len(state.df_indices) + (1 if state.df_idx is not None else 0)}')

    # main content container you will clear/re-render
    root = ui.column().classes('items-stretch w-full max-w-5xl mx-auto gap-3 p-2')

    with ui.footer().classes('justify-center p-2'):
        ui.label('By Daehwa Kim')

    # keep header clock ticking
    ui.timer(1.0, lambda: clock.set_text(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

    def render():
        root.clear()
        with root:
            # no header/footer here; only page content

            if state.df_idx is None:
                ui.markdown('## ✅ Thanks! There are no more items for you right now.')
                return

            step = UI_ORDER[state.ui_idx]
            question = state.get_questionnaire() or ''

            (img_a,img_b) = state.file_path_for_current()
            # prompt = state.prompt_for_current()
            with ui.card().classes('w-full p-4'):
                ui.label("Choose one between two drawings regarding to the question. We realize that creativity probably overlaps other criteria one might consider (for example: aesthetic appeal, or technical execution) but we ask you to rate the artworks solely on the basis of your creativity.").style('font-weight:600; white-space:normal;').classes('break-words')
                ui.label(f"\"{question}\"").style('color: black; font-size: 1.25rem;').classes('break-words')
                ui.separator()

                with ui.row().classes('w-full items-center justify-center gap-4 flex-nowrap'):
                    ui.image(img_a) \
                        .style('width:50%; max-height:70vh; object-fit:contain; background:#000; cursor:pointer;') \
                        .on('click', lambda: (submit_event(state, {'type': 'ab', 'value': 'A'}), render()))
                    ui.element('div').style(
                        'width:2px; align-self:stretch; background:#555; margin:0 12px;'
                    )
                    ui.image(img_b) \
                        .style('width:50%; max-height:70vh; object-fit:contain; background:#000; cursor:pointer') \
                        .on('click', lambda: (submit_event(state, {'type': 'ab', 'value': 'B'}), render()))

            tasks_lbl.set_text(f'Tasks left: {len(state.df_indices) + (1 if state.df_idx is not None else 0)}')

    render()

@ui.page('/')
def index():
    ui.markdown('# HRI Creative Co-Painting\nEnter your participant name to begin.')
    pid = ui.input('Participant name', placeholder='e.g., p1').classes('w-64')
    ui.button(
        'Start',
        on_click=lambda: ui.navigate.to(f'/{(pid.value or "p1").strip()}')
    )

# Expose ASGI app for uvicorn / gunicorn
app = app
if __name__ in {'__main__', '__mp_main__'}:
    ui.run(
        host='0.0.0.0',
        port=16867,
        title='HRI Creative Co-Painting',
        reload=False,
        show=False,
    )
