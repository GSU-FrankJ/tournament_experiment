#!/usr/bin/env python3
"""
Analyze MC-FD results from one_stage_two_players.csv
"""

import csv
import sys

def analyze_mcfd_result(csv_file='results/one_stage_two_players.csv', row_num=29):
    """Analyze a specific MC-FD result row."""
    
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)
    
    if len(rows) < row_num:
        print(f"Error: CSV only has {len(rows)} rows, requested row {row_num}")
        return
    
    # Get header and data row
    header = rows[0]
    data_row = rows[row_num - 1]  # Convert to 0-indexed
    
    # Create dictionary (handle variable column counts)
    result = {}
    for i, col in enumerate(header):
        if i < len(data_row):
            result[col] = data_row[i]
        else:
            result[col] = ''
    
    # Also get any extra columns
    if len(data_row) > len(header):
        # MC-FD specific columns might be after the standard ones
        extra_cols = data_row[len(header):]
        # Try to map known MC-FD columns
        mcfd_cols = [
            'mcfd_final_e1', 'mcfd_final_e2', 'mcfd_iterations',
            'mcfd_sigma1', 'mcfd_sigma2', 'mcfd_delta', 'mcfd_eta',
            'mcfd_samples', 'mcfd_tol', 'mcfd_effort_min', 'mcfd_effort_max'
        ]
        for i, val in enumerate(extra_cols):
            if i < len(mcfd_cols) and val:
                result[mcfd_cols[i]] = val
    
    print("=" * 70)
    print("MC-FD RESULT ANALYSIS")
    print("=" * 70)
    
    print(f"\n📊 BASIC PARAMETERS:")
    print(f"   Stage 1 Weight (w_L): {result.get('stage1_weight', 'N/A')}")
    print(f"   Stage 2 Weight (w_H): {result.get('stage2_weight', 'N/A')}")
    print(f"   Cost Parameter (k): {result.get('k1', 'N/A')}")
    
    print(f"\n🎯 EFFORT RESULTS:")
    e2 = float(result.get('final_stage2_effort', 0))
    weighted = float(result.get('final_weighted_effort', 0))
    print(f"   Final Stage 2 Effort: {e2:.4f}")
    print(f"   Final Weighted Effort: {weighted:.4f}")
    
    # MC-FD specific efforts
    e1_mcfd = result.get('mcfd_final_e1', '')
    e2_mcfd = result.get('mcfd_final_e2', '')
    if e1_mcfd and e2_mcfd:
        print(f"   MC-FD Player 1 Effort (e₁): {float(e1_mcfd):.4f}")
        print(f"   MC-FD Player 2 Effort (e₂): {float(e2_mcfd):.4f}")
        print(f"   Average: {(float(e1_mcfd) + float(e2_mcfd))/2:.4f}")
    
    print(f"\n⚙️  MC-FD HYPERPARAMETERS:")
    if result.get('mcfd_sigma1'):
        print(f"   Noise σ₁: {result.get('mcfd_sigma1')}")
        print(f"   Noise σ₂: {result.get('mcfd_sigma2')}")
    if result.get('mcfd_delta'):
        print(f"   FD Perturbation (δ): {result.get('mcfd_delta')}")
    if result.get('mcfd_eta'):
        print(f"   Learning Rate (η): {result.get('mcfd_eta')}")
    if result.get('mcfd_samples'):
        print(f"   MC Samples (N): {result.get('mcfd_samples')}")
    if result.get('mcfd_iterations'):
        print(f"   Iterations: {result.get('mcfd_iterations')}")
    if result.get('mcfd_tol'):
        print(f"   Tolerance: {result.get('mcfd_tol')}")
    if result.get('mcfd_effort_min'):
        print(f"   Effort Range: [{result.get('mcfd_effort_min')}, {result.get('mcfd_effort_max')}]")
    
    print(f"\n📈 CONVERGENCE:")
    print(f"   Convergence Quality: {result.get('Convergence_Quality', 'N/A')}")
    print(f"   Episodes: {result.get('episodes', 'N/A')}")
    
    print(f"\n🔬 THEORETICAL COMPARISON:")
    theo = result.get('theoretical_stage2_effort', 'N/A')
    gap = result.get('Gap_from_theoretical', 'N/A')
    print(f"   Theoretical Effort: {theo}")
    print(f"   Gap from Theoretical: {gap}")
    if theo == 'nan' or theo == '':
        print(f"   ⚠️  Note: No theoretical benchmark (MC-FD uses Gaussian noise, not uniform)")
    
    print(f"\n💡 INTERPRETATION:")
    print(f"   • Final effort: {e2:.2f}")
    print(f"   • This represents the average effort from both players")
    if e1_mcfd and e2_mcfd:
        e1_val = float(e1_mcfd)
        e2_val = float(e2_mcfd)
        diff = abs(e1_val - e2_val)
        print(f"   • Player 1: {e1_val:.2f}, Player 2: {e2_val:.2f}")
        print(f"   • Difference: {diff:.2f} ({'Symmetric' if diff < 0.1 else 'Asymmetric'})")
    
    quality = result.get('Convergence_Quality', '')
    if quality == 'Poor':
        print(f"\n   ⚠️  Convergence Quality is 'Poor'. Consider:")
        print(f"      - Increasing --mcfd-max-iters (current: {result.get('mcfd_iterations', '?')})")
        print(f"      - Increasing --mcfd-num-samples (current: {result.get('mcfd_samples', '?')})")
        print(f"      - Adjusting --mcfd-eta (current: {result.get('mcfd_eta', '?')})")
        print(f"      - Tightening --mcfd-tol (current: {result.get('mcfd_tol', '?')})")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    row_num = int(sys.argv[1]) if len(sys.argv) > 1 else 29
    analyze_mcfd_result(row_num=row_num)


