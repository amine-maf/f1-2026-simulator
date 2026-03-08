import os
import json

def clean_notebook(path):
    if not os.path.exists(path):
        return
    try:
        with open(path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
            
        changed = False
        for cell in nb.get('cells', []):
            if cell.get('cell_type') == 'markdown':
                source = cell.get('source', [])
                for i, line in enumerate(source):
                    original_line = line
                    for emoji in ['🏎️', '🧪', '🏁', '🚀', '🧠', '📁', '💡', '👉', '⚠️', '📈', '🏎\ufe0f', '🧪']:
                        line = line.replace(emoji + ' ', '').replace(' ' + emoji, '').replace(emoji, '')
                    if line != original_line:
                        source[i] = line
                        changed = True
        
        if changed:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(nb, f, indent=1)
            print(f"Cleaned {path}")
    except Exception as e:
        print(f"Error cleaning {path}: {e}")

def clean_md(path):
    if not os.path.exists(path):
        return
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original = content
    for emoji in ['🏎️', '🧪', '🏁', '🚀', '🧠', '📁', '💡', '👉', '⚠️', '📈', '🏎\ufe0f', '🧪']:
        content = content.replace(emoji + ' ', '').replace(' ' + emoji, '').replace(emoji, '')
        
    if content != original:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Cleaned {path}")

# Run
clean_notebook('02_feature_engineering.ipynb')
clean_notebook('03_race_predictor.ipynb')
clean_md('README.md')
clean_md('/Users/mac/.gemini/antigravity/brain/b38ea19c-45c5-44d9-912f-28ac087544c0/walkthrough.md')
