"""Execute all three notebooks in order and save with outputs."""
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))

notebooks = [
    'Part_A_Data_Preparation.ipynb',
    'Part_B_Analysis.ipynb',
    'Part_C_Actionable_Output_Bonus.ipynb',
]

ep = ExecutePreprocessor(timeout=600, kernel_name='python3')

for nb_file in notebooks:
    print(f"\n{'='*60}")
    print(f"Executing: {nb_file}")
    print(f"{'='*60}")
    try:
        with open(nb_file, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        ep.preprocess(nb, {'metadata': {'path': '.'}})
        # Save executed notebook (overwrite original with outputs)
        with open(nb_file, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        print(f"✅ {nb_file} executed successfully!")
    except Exception as e:
        print(f"❌ Error in {nb_file}: {e}")
        sys.exit(1)

print(f"\n{'='*60}")
print("✅ All notebooks executed successfully!")
print(f"{'='*60}")
