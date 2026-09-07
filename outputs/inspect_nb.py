import json

nb = json.load(open(r'c:\Roshith\Projects\F1-SIS\Lap_Time_Prediction.ipynb', encoding='utf-8'))
out = []
for i, cell in enumerate(nb['cells']):
    ct = cell['cell_type']
    src = ''.join(cell['source'])[:120]
    src_safe = src.encode('ascii', 'replace').decode('ascii')
    out.append(f'{i} {ct}: {src_safe}')

result = '\n'.join(out)
with open(r'c:\Roshith\Projects\F1-SIS\outputs\nb_cells.txt', 'w', encoding='utf-8') as f:
    f.write(result)
print("Written to outputs/nb_cells.txt")
print(result)
