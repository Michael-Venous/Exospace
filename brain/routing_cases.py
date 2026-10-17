"""Deterministic screenshot tasks and independent, executable answer oracles.

Calibration and held-out families differ; no reference answer enters model input.
These are reasoning probes, not a substitute for interaction benchmarks.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from html import escape
from itertools import permutations
import random


@dataclass(frozen=True)
class Case:
    id: str
    split: str
    family: str
    level: int | None
    goal: str
    html: str
    answer: list[str]
    decision: str = "answer"

    def record(self) -> dict:
        return asdict(self)


def screen(title: str, headers: list[str], rows: list[list], note: str = "") -> str:
    cells = lambda row, tag: "<tr>" + "".join(
        f"<{tag}>{escape(str(value))}</{tag}>" for value in row) + "</tr>"
    return ("<!doctype html><meta charset='utf-8'><style>"
            "body{font:20px Arial;color:#18283b;background:#eef2f8;padding:25px}"
            "h1{font-size:30px}table{border-collapse:collapse;background:white}"
            "td,th{padding:9px 17px;border:1px solid #8e9cab;text-align:left}"
            "p{max-width:1150px;line-height:1.4}</style>"
            f"<h1>{escape(title)}</h1><p>{escape(note)}</p><table>"
            + cells(headers, "th") + "".join(cells(row, "td") for row in rows) + "</table>")


def make_cases() -> list[Case]:
    cases = []
    for split in ("calibration", "heldout"):
        for k in range(4):
            rng = random.Random(81713 + k + (100 if split == "heldout" else 0))
            prefix = f"{split}-{k}"
            if split == "calibration":
                rows = [[f"Order {i+21}", rng.choice(["Packed", "Queued", "Shipped"]),
                         f"R-{rng.randrange(100,999)}"] for i in range(7)]
                cases.append(Case(prefix+"-lookup", split, "order_lookup", 0,
                    f"Read the reference code for Order {k+23}. Return only that code in answer.",
                    screen("Order desk", ["Order", "Status", "Reference"], rows), [rows[k+2][2]]))
                actions = ["Open workspace", "Choose project", "Open preferences", "Select export",
                           "Choose CSV", "Confirm local save", "Close preview", "Return home"][:5+k]
                rows = [[f"S{i}", a, f"S{i+1}"] for i,a in enumerate(actions)]
                cases.append(Case(prefix+"-workflow", split, "explicit_workflow", 1,
                    f"Starting at S0, list every action in order to reach S{len(actions)}. "
                    "All actions are already specified on screen. Return action labels in answer.",
                    screen("Workspace procedure", ["Current state", "Action", "Next state"], rows), actions))
                start = 21*60 + rng.randrange(0,120); duration = rng.randrange(65,195)
                delay = rng.randrange(5,50)
                end = (start+delay+duration) % (24*60)
                day = (start+delay+duration) // (24*60)
                rows = [["Original start", f"{start//60:02}:{start%60:02}"],
                        ["Delay in minutes", delay], ["Duration in minutes", duration]]
                cases.append(Case(prefix+"-arithmetic", split, "midnight_calendar", 2,
                    "Apply the delay to the original start, then add duration. Return [end time HH:MM, "
                    "day offset as an integer string]. Day offset 0 means original day, 1 next day.",
                    screen("Calendar update", ["Field", "Value"], rows),
                    [f"{end//60:02}:{end%60:02}", str(day)]))
                matrix = [[rng.randrange(5,96) for _ in range(7)] for _ in range(7)]
                best = min(permutations(range(7)), key=lambda p:(sum(matrix[i][p[i]] for i in range(7)),p))
                cases.append(Case(prefix+"-optimization", split, "worker_assignment", 3,
                    "Assign each worker exactly one different job A through G. Minimize total minutes "
                    "across all seven workers. Return the seven job letters in worker order W1 through W7. "
                    "If totals tie, choose the lexicographically smallest sequence of letters.",
                    screen("Job assignment", ["Worker"]+list("ABCDEFG"),
                           [[f"W{i+1}"]+row for i,row in enumerate(matrix)], "Cells are time in minutes."),
                    ["ABCDEFG"[j] for j in best]))
            else:
                rows = [[f"Sensor {i+1}", f"{rng.randrange(10,90)}%", f"Zone {chr(65+i)}"] for i in range(7)]
                cases.append(Case(prefix+"-lookup", split, "sensor_lookup", 0,
                    f"What percentage is shown for Sensor {k+2}? Return the percentage including % in answer.",
                    screen("Sensor readings", ["Sensor", "Level", "Location"], rows), [rows[k+1][1]]))
                path = ["Tools", "Export", ["Reports","Charts","Tables","Logs"][k], "CSV"]
                rows = [["Home", "Tools", "Tools menu"], ["Home", "Help", "Help panel"],
                        ["Tools menu", "Export", "Export menu"], ["Tools menu", "Import", "Import panel"],
                        ["Export menu", path[2], "Format menu"], ["Format menu", "CSV", "CSV confirmation"],
                        ["Format menu", "PDF", "PDF confirmation"]]
                cases.append(Case(prefix+"-workflow", split, "menu_path", 1,
                    "Starting from Home, list the menu labels to reach CSV confirmation in order. "
                    "The menu transitions are shown. Return labels only in answer.",
                    screen("Menu guide", ["Location", "Menu label", "Reveals"], rows), path))
                quantities = [rng.randrange(2,8) for _ in range(3)]
                prices = [rng.randrange(100,2000) for _ in range(3)]
                discount = rng.choice([10,20,25]); shipping = rng.randrange(100,600)
                # Integer cents, round discounted subtotal half up before shipping.
                total = (sum(q*p for q,p in zip(quantities,prices))*(100-discount)+50)//100 + shipping
                rows = [[f"Item {i+1}", q, f"{p//100}.{p%100:02}"] for i,(q,p) in enumerate(zip(quantities,prices))]
                cases.append(Case(prefix+"-arithmetic", split, "cart_total", 2,
                    "Calculate the total: quantity times unit price, apply the discount to the subtotal, "
                    "round to cents (half up), then add shipping. No tax. Return one decimal amount with "
                    "exactly two decimal places and no currency symbol.",
                    screen("Local cart", ["Product", "Quantity", "Unit price"], rows,
                           f"Discount {discount}%. Shipping {shipping//100}.{shipping%100:02}. No tax."),
                    [f"{total//100}.{total%100:02}"]))
                items = [(rng.randrange(4,20),rng.randrange(2,12),rng.randrange(10,80)) for _ in range(12)]
                budget, hours = 56+k*2, 28+k
                candidates=[]
                for mask in range(1<<len(items)):
                    chosen=tuple(i for i in range(len(items)) if mask>>i&1)
                    cost=sum(items[i][0] for i in chosen); time=sum(items[i][1] for i in chosen)
                    if cost<=budget and time<=hours:
                        candidates.append((-sum(items[i][2] for i in chosen),
                                           tuple(f"P{i+1:02}" for i in chosen)))
                best=min(candidates)[1]
                cases.append(Case(prefix+"-optimization", split, "project_selection", 3,
                    "Select the subset of projects with maximum total value while respecting both budget "
                    "and hours. Each can be chosen at most once. Return selected project IDs in ascending "
                    "order. For equal total value choose the lexicographically smallest sorted ID list.",
                    screen("Project portfolio", ["Project", "Cost", "Hours", "Value"],
                           [[f"P{i+1:02}",*row] for i,row in enumerate(items)],
                           f"Budget at most {budget}; hours at most {hours}. Values add independently."), list(best)))
        if split == "calibration":
            for k in range(2):
                cases.append(Case(f"{split}-{k}-ambiguous", split, "duplicate_controls", None,
                    "Click Continue. Identify the intended workspace, or ask if the request does not specify it.",
                    screen("Workspaces", ["Workspace", "Button"],
                           [["Personal" if k==0 else "Research","Continue"], ["Team" if k==0 else "Teaching","Continue"]]),
                    [], "ask_user"))
        else:
            for k in range(2):
                cases.append(Case(f"{split}-{k}-ambiguous", split, "missing_preference", None,
                    "Select the file I prefer. Return its filename or ask if my preference is not provided.",
                    screen("Files", ["Filename", "Size"], [[f"draft-{k}-A.txt","16 KB"],[f"draft-{k}-B.txt","16 KB"]],
                           "No file is selected. No preference has been recorded."), [], "ask_user"))
    return cases
