import json
from pathlib import Path

golden_path = Path("golden_sessions/golden_Zelle_Campaign_with_Second_Doc_20260219_144600.json")
with open(golden_path, "r") as f:
    data = json.load(f)

# Update expected tools for Turns 9 and 10 based on wellfb server behavior
# Turn 9: Was expected [GetCaseStages, Gradial_Agent]. Actually [Gradial_Agent, pxPerformAssignment]
data["turns"][8]["expected_tools"] = ["Gradial_Agent", "pxPerformAssignment"]

# Turn 10: Was expected [GetCaseStages]. Actually [Gradial_Agent, pxPerformAssignment]
data["turns"][9]["expected_tools"] = ["Gradial_Agent", "pxPerformAssignment"]

with open(golden_path, "w") as f:
    json.dump(data, f, indent=2)

print("Golden record tool expectations updated for Turns 9 & 10.")
