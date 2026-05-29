import json
from pathlib import Path

golden_path = Path("golden_sessions/golden_Zelle_Campaign_with_Second_Doc_20260219_144600.json")
with open(golden_path, "r") as f:
    data = json.load(f)

# Update inputs to be more likely to satisfy the current agent's "Yes/No" requirement
updates = {
    3: "Yes",
    4: "Yes",
    6: "Move forward with this content",
    7: "Move forward",
    8: "Move forward with this audience",
    9: "Yes",
    10: "Move forward with this audience"
}

for turn in data["turns"]:
    t_num = turn["turn"]
    if t_num in updates:
        print(f"Updating Turn {t_num}: '{turn['input']}' -> '{updates[t_num]}'")
        turn["input"] = updates[t_num]

with open(golden_path, "w") as f:
    json.dump(data, f, indent=2)

print("Golden record updated with 'Yes' inputs for better progression.")
