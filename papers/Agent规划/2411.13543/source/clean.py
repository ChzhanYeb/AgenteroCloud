import glob

for filepath in glob.glob('**/*.tex', recursive=True):
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read()
    content = content.replace('’', "'").replace('‘', "'")
    content = content.replace('“', '"').replace('”', '"')
    content = content.replace('–', '-').replace('—', '-')
    content = content.replace('…', '...')
    with open(filepath, 'w', encoding='ascii', errors='ignore') as file:
        file.write(content)
print("All .tex files converted to ASCII.")