import json

def convert_ipynb_to_py(ipynb_path, py_path):
    with open(ipynb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    code_blocks = []
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = cell.get('source', '')
            if isinstance(source, list):
                source = "".join(source)
            # Remove any jupyter magic commands like !pip or %matplotlib
            filtered_source = "\n".join([line for line in source.split('\n') if not line.strip().startswith('!') and not line.strip().startswith('%')])
            filtered_source = filtered_source.replace('plt.show()', 'plt.close()')
            code_blocks.append(filtered_source)
    
    with open(py_path, 'w', encoding='utf-8') as f:
        f.write("\n\n".join(code_blocks))
        
if __name__ == "__main__":
    convert_ipynb_to_py('modelbuild.ipynb', 'modelbuild.py')
    print("Notebook successfully converted to modelbuild.py")
