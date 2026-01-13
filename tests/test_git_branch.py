import pytest
import subprocess
import os

@pytest.mark.skip("not enforcing these branch names; doesn't work with a shallow clone")
def test_git_branch_contains_right_changes():
    files = subprocess.check_output(["git", "diff", "--name-only", "main"], text=True).split("\n")
    branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
    # Reverting a merge from the UI creates non-standard branch names
    kind, name = branch.split("/",1)
    psr_files = []
    other_files = []
    for f in files:
        d, b = os.path.dirname(f), os.path.basename(f)
        if d == "configs" and (b.startswith("J") or b.startswith("B")):
            psr_files.append(f)
        elif d == "results" or d == "results/archive":
            psr_files.append(f)
        else:
            other_files.append(f)
    if kind == "psr":
        psr_name = name.split("/")[0].upper()
        for f in psr_files:
            assert os.path.basename(f).startswith(psr_name)
        assert not other_files
    elif kind == "feature" or kind == "hotfix":
        assert not psr_files
    else:
        raise ValueError(f"Unrecognized branch name {branch}")


