import re

content = open('bot/web_server.py', 'r', encoding='utf-8').read()

# Find HTML template
start = content.find('HTML_TEMPLATE = """')
end = content.find('"""', start + 20)
html = content[start:end]

# Extract JS
js_start = html.find('<script>')
js_end = html.find('</script>')
js = html[js_start:js_end]

# Split JS into statements by semicolons, respecting strings
statements = []
current = ''
in_string = False
string_char = None
escape = False

for char in js:
    if escape:
        current += char
        escape = False
        continue
    if char == '\\':
        current += char
        escape = True
        continue
    if not in_string and char in '"\'`':
        in_string = True
        string_char = char
        current += char
    elif in_string and char == string_char:
        in_string = False
        string_char = None
        current += char
    elif not in_string and char == ';':
        current = current.strip()
        if current:
            statements.append(current)
        current = ''
    else:
        current += char

current = current.strip()
if current:
    statements.append(current)

# Check each statement for balanced parens/braces/brackets
for i, stmt in enumerate(statements):
    parens = 0
    braces = 0
    brackets = 0
    in_str = False
    str_char = None
    escape = False
    
    for char in stmt:
        if escape:
            escape = False
            continue
        if char == '\\':
            escape = True
            continue
        if not in_str and char in '"\'`':
            in_str = True
            str_char = char
        elif in_str and char == str_char:
            in_str = False
            str_char = None
        elif not in_str:
            if char == '(': parens += 1
            elif char == ')': parens -= 1
            elif char == '{': braces += 1
            elif char == '}': braces -= 1
            elif char == '[': brackets += 1
            elif char == ']': brackets -= 1
    
    if parens != 0 or braces != 0 or brackets != 0:
        print(f'STATEMENT {i+1} UNBALANCED: parens={parens}, braces={braces}, brackets={brackets}')
        print(stmt[:300])
        print('---')

print(f'Total statements checked: {len(statements)}')
