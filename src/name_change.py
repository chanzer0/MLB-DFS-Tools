import fileinput
import os

try:
    projection_path = os.path.join(
        os.path.dirname(__file__), "../dk_data/projections.csv"
    )
    with fileinput.FileInput(projection_path, inplace=True) as file:
        for line in file:
            print(line.replace("SS/OF", "OF/SS").replace("C/1B", "1B/C"), end="")
except:
    print("DK Projections failed to rename")
    pass
