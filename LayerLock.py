import os
import json
import csv
import argparse
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Iterable, Set, Optional
from collections import defaultdict

from Bio.PDB import PDBParser

# =====================================================
# Config data models (RUN-CENTRIC, one PDB per run)
# =====================================================

@dataclass
class LockSpec:
    """Lock seed group for a run. layers=0 means include seeds directly (no expansion)."""
    residues: List[str]
    layers: int = 0
    cutoff: float = 5.0

@dataclass
class RunSpec:
    name: str                         # produces fix_<name>.json
    pdb_id: str
    chain: str = "A"                  # chain to process
    include_resnames: List[str] = None # optional include by resname
    default_chain: str = "A"
    locks: List[LockSpec] = None       # lock seed groups

    def __post_init__(self):
        if self.include_resnames is None:
            self.include_resnames = []
        if self.locks is None:
            self.locks = []

# =====================================================
# Residue helpers
# =====================================================

STANDARD_RESIDUES = {
    "ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS", "LEU",
    "MET", "ASN", "PRO", "GLN", "ARG", "SER", "THR", "VAL", "TRP", "TYR"
}

def is_standard_residue(res) -> bool:
    return res.resname.strip().upper() in STANDARD_RESIDUES

def format_residue_id(chain_id: str, resseq: int, icode: str) -> str:
    icode = (icode or "").strip()
    return f"{chain_id}{resseq}{icode}" if icode else f"{chain_id}{resseq}"

def parse_residue_token(tok: str, default_chain: str) -> Tuple[str, int, str]:
    s = tok.strip()
    if not s:
        raise ValueError("Empty residue token")
    i = 0
    while i < len(s) and not s[i].isdigit():
        i += 1
    chain = s[:i] or default_chain
    rest = s[i:]
    if not rest:
        raise ValueError(f"Invalid residue token: '{tok}'")
    if rest[-1].isalpha():
        icode = rest[-1]
        num_part = rest[:-1]
    else:
        icode = ""
        num_part = rest
    resseq = int(num_part)
    return chain, resseq, icode

def collect_standard_residues(model, chain_id: str) -> List[Tuple[object, str]]:
    if chain_id not in model:
        return []
    return [
        (res, format_residue_id(chain_id, res.id[1], res.id[2]))
        for res in model[chain_id]
        if is_standard_residue(res)
    ]

def get_res_by_tokens(model, tokens: Iterable[str], default_chain: str) -> List[object]:
    out = []
    print('asfhbfbkfs',tokens)
    for tok in (tokens or []):
        try:
            chain, resseq, icode = parse_residue_token(tok, default_chain)
        except Exception:
            continue
        # Enforce per-run chain scoping: ignore tokens that explicitly name a different chain
        if chain and chain != default_chain:
            continue
        if default_chain not in model:
            continue
        for res in model[default_chain]:
            if is_standard_residue(res) and res.id[1] == resseq and (res.id[2] or "").strip() == (icode or "").strip():
                out.append(res)
                break
    return out


def get_residues_by_resname(model, chain_id: str, resnames: Iterable[str]) -> Set[str]:
    if chain_id not in model:
        return set()
    resnames_upper = {r.upper() for r in (resnames or [])}
    return {
        format_residue_id(chain_id, res.id[1], res.id[2])
        for res in model[chain_id]
        if res.resname.strip().upper() in resnames_upper
    }

def get_neighbors(model, chain_id: str, current_residues: Iterable[object], cutoff: float) -> Set[object]:
    if chain_id not in model:
        return set()
    neighbors = set()
    current_atoms = [atom for res in (current_residues or []) for atom in res.get_atoms()]
    if not current_atoms:
        return set()
    for res in model[chain_id]:
        if not is_standard_residue(res):
            continue
        for atom in res.get_atoms():
            if any(atom - ref_atom < cutoff for ref_atom in current_atoms):
                neighbors.add(res)
                break
    return neighbors

# =====================================================
# I/O helpers
# =====================================================

def _normalize_residues_field(val) -> List[str]:
    """
    Accept list or string for residues. Strings may contain:
      - list separators: ',', ';', whitespace
      - hyphen-separated *lists* with no other separators (e.g. "10-12-45-62" or "A10-A46-A90")
      - simple ranges when used as a single token between separators: "10-15", "A10-A15"
        * If both sides carry a chain letter, they must match (e.g., "A10-A15").
        * Insertion codes are NOT supported inside ranges (use explicit tokens instead).
    Rules:
      • If the string has any of comma/semicolon/whitespace, we split on those first.
        Each resulting piece that has exactly one hyphen and both sides numeric is expanded as a range.
      • If the string has no separators except hyphens, we treat hyphens as list delimiters (no range expansion).
    """
    def expand_range_piece(piece: str) -> List[str]:
        piece = piece.strip()
        if not piece or '-' not in piece or piece.count('-') != 1:
            return [piece] if piece else []
        left, right = piece.split('-', 1)
        # parse left
        li = 0
        while li < len(left) and not left[li].isdigit():
            li += 1
        lchain = left[:li]
        lnum = left[li:]
        # parse right
        ri = 0
        while ri < len(right) and not right[ri].isdigit():
            ri += 1
        rchain = right[:ri]
        rnum = right[ri:]
        if not lnum.isdigit() or not rnum.isdigit():
            return [piece]
        chain = lchain or rchain
        if lchain and rchain and lchain.upper() != rchain.upper():
            return [piece]
        start = int(lnum)
        end = int(rnum)
        if start > end:
            start, end = end, start
        return [f"{chain}{i}" if chain else str(i) for i in range(start, end + 1)]

    if isinstance(val, list):
        return [str(x).strip() for x in val if str(x).strip()]
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return []
        # If there are commas/semicolons/whitespace, split on those first
        if any(sep in s for sep in [',', ';', '	', '\n', ' ']):
            for sep in [',', ';', '	', '\n']:
                s = s.replace(sep, ' ')
            pieces = [p for p in s.split() if p]
            out: List[str] = []
            for p in pieces:
                if '-' in p and p.count('-') == 1:
                    out.extend(expand_range_piece(p))
                else:
                    out.append(p)
            return out
        # Otherwise, no separators besides hyphens: interpret as hyphen-delimited list
        return [t for t in s.split('-') if t]
    if val is None:
        return []
    return [str(val).strip()]

def read_config_from_json(path: str) -> List[RunSpec]:
    with open(path) as f:
        data = json.load(f)
    runs: List[RunSpec] = []
    for r in data.get("runs", []):
        pdb_id=r.get("pdb_id")
        if isinstance(pdb_id, str):
            runs.append(RunSpec(
                name=r.get("name"),
                pdb_id=r.get("pdb_id"),
                chain=r.get("chain", "A"),
                include_resnames=list(r.get("include_resnames", []) or []),
                default_chain=(r.get("default_chain") or r.get("chain", "A")),
                locks=[
                LockSpec(
                    residues=_normalize_residues_field(l.get("residues")),
                    layers=int(l.get("layers", 0)),
                    cutoff=float(l.get("cutoff", 5.0)),
                )
                for l in (r.get("locks") or [])
            ]
            ))
        elif isinstance(pdb_id, list):
            for tmppdbid in pdb_id:
                runs.append(RunSpec(
                    name=r.get("name"),
                    pdb_id=tmppdbid,
                    chain=r.get("chain", "A"),
                    include_resnames=list(r.get("include_resnames", []) or []),
                    default_chain=(r.get("default_chain") or r.get("chain", "A")),
                    locks=[
                    LockSpec(
                        residues=_normalize_residues_field(l.get("residues")),
                        layers=int(l.get("layers", 0)),
                        cutoff=float(l.get("cutoff", 5.0)),
                    )
                    for l in (r.get("locks") or [])
                ]
                ))
    return runs

# =====================================================
# Core driver
# =====================================================

def run(
    runs: List[RunSpec],
    pdb_dir: str,
    out_prefix: str = "fix_",
    summary_csv: str = "summary.csv",
    write_run_defs: Optional[str] = "run_definitions.json",
    debug: bool = False,
):
    parser = PDBParser(QUIET=True)
    def dprint(*args, **kwargs):
        if debug:
            print(*args, **kwargs)

    group_outputs: Dict[str, Dict[str, str]] = defaultdict(dict)
    summary_rows: List[Dict[str, object]] = []
    run_definitions = {r.name: asdict(r) for r in runs}

    for run_spec in runs:
        dprint(f"\n[RUN] {run_spec.name} | PDB={run_spec.pdb_id} | chain={run_spec.chain}")
        pdb_path = os.path.join(pdb_dir, f"{run_spec.pdb_id}.pdb")
        if not os.path.exists(pdb_path):
            print(f"[WARN] Missing PDB file: {pdb_path}")
            continue
        structure = parser.get_structure(run_spec.pdb_id, pdb_path)
        dprint(f"  Loaded structure: {pdb_path}")
        model = structure[0]
        chain_id = run_spec.chain or run_spec.default_chain

        all_pairs = collect_standard_residues(model, chain_id)
        dprint(f"  Standard residues on chain {chain_id}: {len(all_pairs)}")

        all_ids = {rid for _, rid in all_pairs}
        total_residues_count = len(all_ids)

        expanded_ids: Set[str] = set()
        
        for i, lock in enumerate(run_spec.locks or []):
            dprint(f"    [LOCK {i+1}] raw residues: {lock.residues} | layers={lock.layers} | cutoff={lock.cutoff}")
        for lock in (run_spec.locks or []):
            seed_residues = get_res_by_tokens(model, lock.residues, chain_id)
            dprint(f"      seeds matched on chain {chain_id}: {len(seed_residues)}")
            seen = set(seed_residues)
            current = set(seed_residues)
            for layer in range(lock.layers):
                nbrs = get_neighbors(model, chain_id, current, float(lock.cutoff))
                new = nbrs - seen
                dprint(f"        layer {layer+1}: +{len(new)} (cumulative {len(seen)})")
                seen.update(new)
                current = new
            expanded_ids |= {format_residue_id(chain_id, res.id[1], res.id[2]) for res in seen}
            dprint(f"      expanded set size after LOCK {i+1}: {len(expanded_ids)}")
        include_by_name = get_residues_by_resname(model, chain_id, run_spec.include_resnames)
        if run_spec.include_resnames:
            dprint(f"  include_resnames {run_spec.include_resnames} adds: {len(include_by_name)}")
        final_ids = (expanded_ids | include_by_name) & all_ids
        dprint(f"  FINAL locked={len(final_ids)} / total={len(all_ids)}")
        final_ids_sorted = sorted(final_ids)

        group_outputs[run_spec.name][f"{run_spec.pdb_id}.pdb"] = " ".join(final_ids_sorted)
        summary_rows.append({
            "Run": run_spec.name,
            "PDB ID": run_spec.pdb_id,
            "Total_Residues": total_residues_count,
            "Locked": len(final_ids_sorted),
            "Unlocked": max(0, total_residues_count - len(final_ids_sorted)),
        })

    for run_name, group_dict in group_outputs.items():
        with open(f"{out_prefix}{run_name}.json", "w") as f:
            json.dump(group_dict, f, indent=2)

    if summary_csv:
        with open(summary_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["Run", "PDB ID", "Total_Residues", "Locked", "Unlocked"])
            w.writeheader()
            w.writerows(summary_rows)

    if write_run_defs:
        with open(write_run_defs, "w") as f:
            json.dump(run_definitions, f, indent=2)

    print("Done! Created:")
    for name in group_outputs:
        print(f"  {out_prefix}{name}.json")
    if summary_csv:
        print(f"  {summary_csv}")
    if write_run_defs:
        print(f"  {write_run_defs}")

# =====================================================
# CLI
# =====================================================

def main():
    ap = argparse.ArgumentParser(description="ProteinMPNN locking with run-centric JSON: one pdb+chain per run, multiple lock seed groups.")
    ap.add_argument("--config", default=False, help="Path to JSON with runs")

    ap.add_argument("--pdbid", default=False, help="Name of pdb file to generate constraints for. No extension")
    ap.add_argument("--runname", default=False, help="Name of locking strategy/run")
    ap.add_argument("--chain", default="A", help="Chain of pdb file")
    ap.add_argument("--enzyme", action="store_true", help="Will lock all ionizable residues from design (CDEHKRSY")
    ap.add_argument("--lock_inputres", default="", help="PDB residue positions of residues to be locked. Can be used with layers to generate full locklist")
    ap.add_argument("--layers_num", default=2, help="Number of layers, from inputres, to consider for locking")    
    ap.add_argument("--layers_cutoff", default=5, help="Path to JSON with runs")

    ap.add_argument("--pdb_dir", default=".", help="Directory containing PDB files")
    ap.add_argument("--out_prefix", default="fix_", help="Prefix for per-run JSON outputs")
    ap.add_argument("--summary_csv", default="summary.csv", help="Summary CSV path (or empty to skip)")
    ap.add_argument("--write_run_defs", default="run_definitions.json", help="Path to echo run defs JSON (or empty to skip)")
    ap.add_argument("--debug", action="store_true", help="Verbose diagnostics: show seed parsing, per-layer growth, and counts")
    args = ap.parse_args()

    if args.config:
        runs = read_config_from_json(args.config)
    elif not args.runname:
        args.runname = args.pdbid
    elif args.pdbid and args.enzyme:
        runs = [RunSpec(name=args.runname, 
                            pdb_id=args.pdbid, 
                            chain=args.chain, 
                            include_resnames=["CYS","ASP","GLU","HIS","LYS","ARG","SER","TYR"],
                            locks=[LockSpec(residues=_normalize_residues_field(args.lock_inputres), 
                                            layers=args.layers_num,
                                            cutoff=args.layers_cutoff)
                                    ]
                            )]
    else:
        print("You need to designate a config file or a pdbid")
        exit()
    print(runs[0].locks)
    run(
        runs=runs,
        pdb_dir=args.pdb_dir,
        out_prefix=args.out_prefix,
        summary_csv=args.summary_csv or None,
        write_run_defs=args.write_run_defs or None,
        debug=args.debug,
    )

if __name__ == "__main__":
    main()

"""
Example JSON config:
{
  "runs": [
    {
      "name": "example_run_1",
      "pdb_id": "1ABC",
      "chain": "A",
      "include_resnames": ["CYS", "ASP"],
      "locks": [
        {"residues": ["A10", "A12"], "layers": 2, "cutoff": 5.0},
        {"residues": "A45-A47",       "layers": 1, "cutoff": 5.0},
        {"residues": "A60 A62",       "layers": 0, "cutoff": 0.0},
        {"residues": "10-12, 20  22", "layers": 0, "cutoff": 0.0}
      ]
    },
    {
      "name": "example_run_2",
      "pdb_id": "2XYZ",
      "chain": "B",
      "include_resnames": [],
      "locks": [
        {"residues": "B15-B17", "layers": 3, "cutoff": 6.0}
      ]
    }
  ]
}
"""
