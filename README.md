# Jigsaw Puzzle Solver

A tool to find adjacent pairs of white jigsaw puzzle pieces based on their shape.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Run the solver:

```bash
python3 solver.py
```

This will generate:
- `matches.txt`: A list of potential matches sorted by score (lower is better).
- `match_vis.jpg`: A visualization of the top match.

To verify side numbers on a specific piece:

```bash
python3 label_sides.py
```
(Edit the script to change the target filename)

## Side Numbering

Sides are numbered 0-3 counter-clockwise from the top-left:
- **0**: Left
- **1**: Bottom
- **2**: Right
- **3**: Top
