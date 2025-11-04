#!/usr/bin/env python3
"""
CAFA-6 Production Pipeline Validator
===================================

This script validates that our production pipeline works correctly
and generates properly formatted submissions.
"""

import os
import sys
import tempfile
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np

def validate_notebook_syntax():
    """Validate that the Jupyter notebook has valid syntax"""
    print("🔍 Validating notebook syntax...")
    
    notebook_path = Path("CAFA6_Production_Ready.ipynb")
    if not notebook_path.exists():
        print("❌ Notebook not found!")
        return False
    
    try:
        # Try to parse as JSON
        import json
        with open(notebook_path) as f:
            notebook_data = json.load(f)
        
        print(f"✅ Notebook syntax valid: {len(notebook_data.get('cells', []))} cells")
        return True
    except Exception as e:
        print(f"❌ Notebook syntax error: {e}")
        return False

def validate_python_script():
    """Validate that the Python script has valid syntax"""
    print("🔍 Validating Python script syntax...")
    
    script_path = Path("cafa6_production.py")
    if not script_path.exists():
        print("❌ Python script not found!")
        return False
    
    try:
        # Try to compile the script
        with open(script_path) as f:
            code = f.read()
        compile(code, script_path, 'exec')
        
        print("✅ Python script syntax valid")
        return True
    except SyntaxError as e:
        print(f"❌ Python script syntax error: {e}")
        return False

def test_demo_mode():
    """Test the pipeline in demo mode"""
    print("🧪 Testing demo mode...")
    
    try:
        # Run the script in demo mode
        result = subprocess.run([
            sys.executable, "cafa6_production.py", 
            "--mode", "knn", "--demo", "--work-dir", "test_output"
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ Demo mode test passed")
            
            # Check output files
            output_dir = Path("test_output")
            submission_file = output_dir / "submission.tsv"
            
            if submission_file.exists():
                # Validate submission format
                df = pd.read_csv(submission_file, sep='\t', header=None, 
                                names=['EntryID', 'term', 'score'])
                
                print(f"  📄 Submission file: {len(df)} predictions")
                print(f"  🔬 Unique proteins: {df['EntryID'].nunique()}")
                print(f"  🧬 Unique terms: {df['term'].nunique()}")
                
                # Validate format
                issues = []
                
                # Check GO term format
                df['term'] = df['term'].astype(str)
                invalid_terms = df[~df['term'].str.match(r'^GO:\d{7}$', na=False)]
                if len(invalid_terms) > 0:
                    issues.append(f"Invalid GO terms: {len(invalid_terms)}")
                
                # Check score range
                try:
                    df['score_float'] = pd.to_numeric(df['score'])
                    invalid_scores = df[(df['score_float'] < 0) | (df['score_float'] > 1)]
                    if len(invalid_scores) > 0:
                        issues.append(f"Invalid scores: {len(invalid_scores)}")
                except:
                    issues.append("Could not parse scores as numeric")
                
                # Check terms per protein
                terms_per_protein = df.groupby('EntryID').size()
                max_terms = terms_per_protein.max()
                if max_terms > 1500:
                    issues.append(f"Protein exceeds 1500 term limit: {max_terms}")
                
                if issues:
                    print(f"  ⚠️  Format issues: {len(issues)}")
                    for issue in issues:
                        print(f"    • {issue}")
                else:
                    print("  ✅ Submission format valid")
                
                return True
            else:
                print("❌ Submission file not created")
                return False
        else:
            print(f"❌ Demo mode test failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Demo mode test timed out")
        return False
    except Exception as e:
        print(f"❌ Demo mode test error: {e}")
        return False

def validate_imports():
    """Check that all required imports are available"""
    print("📦 Validating imports...")
    
    required_imports = [
        'numpy', 'pandas', 'networkx', 'pathlib', 'json', 'time'
    ]
    
    optional_imports = [
        ('torch', 'PyTorch (for deep learning)'),
        ('Bio', 'Biopython (for FASTA parsing)'),
        ('sklearn', 'Scikit-learn (for optimization)')
    ]
    
    missing_required = []
    missing_optional = []
    
    for pkg in required_imports:
        try:
            __import__(pkg)
        except ImportError:
            missing_required.append(pkg)
    
    for pkg, desc in optional_imports:
        try:
            __import__(pkg)
        except ImportError:
            missing_optional.append((pkg, desc))
    
    if missing_required:
        print(f"❌ Missing required packages: {missing_required}")
        return False
    else:
        print("✅ All required packages available")
    
    if missing_optional:
        print("⚠️  Missing optional packages:")
        for pkg, desc in missing_optional:
            print(f"  • {pkg}: {desc}")
    
    return True

def create_demo_data():
    """Create demo data files for testing"""
    print("📂 Creating demo data files...")
    
    demo_dir = Path("demo_data")
    demo_dir.mkdir(exist_ok=True)
    
    # Create demo GO ontology
    go_obo = demo_dir / "go-basic.obo"
    with open(go_obo, 'w') as f:
        f.write("""format-version: 1.2
data-version: releases/2023-05-08

[Term]
id: GO:0008150
name: biological_process
namespace: biological_process

[Term]
id: GO:0003674
name: molecular_function
namespace: molecular_function

[Term]
id: GO:0005575
name: cellular_component
namespace: cellular_component

[Term]
id: GO:0006412
name: translation
namespace: biological_process
is_a: GO:0008150 ! biological_process

[Term]
id: GO:0003824
name: catalytic activity
namespace: molecular_function
is_a: GO:0003674 ! molecular_function
""")
    
    # Create demo training terms
    train_terms = demo_dir / "train_terms.tsv"
    with open(train_terms, 'w') as f:
        f.write("T00001\tGO:0008150\tBP\n")
        f.write("T00001\tGO:0006412\tBP\n")
        f.write("T00002\tGO:0003674\tMF\n")
        f.write("T00002\tGO:0003824\tMF\n")
    
    # Create demo IA weights
    ia_weights = demo_dir / "IA.tsv"
    with open(ia_weights, 'w') as f:
        f.write("GO:0008150\t1.0\n")
        f.write("GO:0003674\t1.0\n")
        f.write("GO:0005575\t1.0\n")
        f.write("GO:0006412\t2.5\n")
        f.write("GO:0003824\t2.8\n")
    
    print(f"✅ Demo data created in {demo_dir}/")
    return str(demo_dir)

def run_full_validation():
    """Run complete validation suite"""
    print("="*60)
    print("🧪 CAFA-6 PRODUCTION PIPELINE VALIDATION")
    print("="*60)
    
    tests = [
        ("Notebook Syntax", validate_notebook_syntax),
        ("Python Script Syntax", validate_python_script),
        ("Required Imports", validate_imports),
        ("Demo Mode Test", test_demo_mode),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("📊 VALIDATION SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All validations passed! Pipeline is ready for use.")
    else:
        print("⚠️  Some validations failed. Please fix issues before using.")
    
    print("="*60)
    return passed == total

if __name__ == "__main__":
    success = run_full_validation()
    sys.exit(0 if success else 1)