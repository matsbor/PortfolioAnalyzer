#!/usr/bin/env python3
"""
Quick fix for indentation error in v3_final
"""

import sys
from pathlib import Path

def fix_indentation(filename):
    """Fix indentation issues"""
    
    print(f"🔧 Fixing indentation in {filename}...")
    
    if not Path(filename).exists():
        print(f"❌ File not found: {filename}")
        return False
    
    # Read file
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    print(f"✅ Read {len(lines)} lines")
    
    # Fix common indentation issues around line 2037
    fixed_lines = []
    for i, line in enumerate(lines, 1):
        # Check for problematic patterns
        if 'if INSTITUTIONAL_V3_AVAILABLE:' in line:
            # Make sure it has proper indentation (12 spaces for inside function)
            if line.startswith('              if'):
                # This is too much indentation
                line = '            ' + line.lstrip()
                print(f"   Fixed line {i}: reduced indentation")
        
        fixed_lines.append(line)
    
    # Backup
    backup = filename.replace('.py', '_broken_backup.py')
    with open(backup, 'w') as f:
        f.writelines(lines)
    print(f"   📦 Backup: {backup}")
    
    # Write fixed
    with open(filename, 'w') as f:
        f.writelines(fixed_lines)
    print(f"   ✅ Fixed and saved")
    
    return True

if __name__ == '__main__':
    filename = 'alpha_miner_pro_v3_final.py'
    
    if fix_indentation(filename):
        print("\n✅ Fix applied! Try running again:")
        print(f"  streamlit run {filename}")
    else:
        print("\n❌ Fix failed. Recommend using v2 instead:")
        print("  streamlit run alpha_miner_institutional_v2.py")
