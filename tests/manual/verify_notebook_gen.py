"""Verify notebook generation is correct after bug fixes."""
import tempfile, ast
from pathlib import Path
import pandas as pd, nbformat

tmp = Path(tempfile.mkdtemp())
csv_path = tmp / 'data.csv'
pd.DataFrame({
    'age': [25, 30, 35, 40],
    'salary': [50000, 60000, 70000, 80000],
    'label': ['yes', 'no', 'yes', 'no'],
}).to_csv(csv_path, index=False)
(tmp / 'notebooks').mkdir()

from backend.services.notebook_gen import generate_notebook

nb_path = generate_notebook(
    experiment_id='test-fix-001',
    dataset_path=csv_path,
    full_config={
        'columns': {
            'age':    {'type': 'numerical',   'strategy': 'standardize', 'imputation': 'none', 'imputation_fill_value': None, 'is_target': False},
            'salary': {'type': 'numerical',   'strategy': 'normalize',   'imputation': 'none', 'imputation_fill_value': None, 'is_target': False},
            'label':  {'type': 'categorical', 'strategy': 'label',       'imputation': 'none', 'imputation_fill_value': None, 'is_target': True},
        },
        'outlier_treatment': {'method': 'none', 'threshold': 1.5},
        'feature_selection': {'method': 'none', 'k': 10, 'score_func': 'f_classif', 'variance_threshold': 0.0},
        'class_balancing':   {'method': 'none'},
    },
    models_config=[{'name': 'LogisticRegression', 'parameters': {'C': 1, 'max_iter': 200}}],
    problem_type='classification',
    tuning_config_json=None,
    storage_dir=tmp,
)

nb = nbformat.read(str(nb_path), as_version=4)
print(f'Notebook has {len(nb.cells)} cells')

# 1. All code cells must parse as valid Python
all_code = '\n'.join(cell.source for cell in nb.cells if cell.cell_type == 'code')
try:
    ast.parse(all_code)
    print('PASS: all cell code is syntactically valid Python')
except SyntaxError as e:
    print(f'FAIL SyntaxError: {e}')
    raise

train_cell = nb.cells[-1].source

# 2. experiment_id must be embedded as the real value
assert 'test-fix-001' in train_cell, f'experiment_id not embedded: {train_cell[:300]}'
print('PASS: experiment_id correctly embedded')

# 3. joblib filename must be a plain string, not an f-string with class reference
assert "joblib.dump(model, 'LogisticRegression_model.joblib')" in train_cell, \
    f"joblib filename wrong in: {train_cell}"
print("PASS: joblib filename is correct string literal")

print('\nAll checks passed!')
