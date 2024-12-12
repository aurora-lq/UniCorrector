# output "error type\ vaild SMILES\ invaild SMILES"


infile = ""
outfile = ""

error_categories = {
    "Aromaticity Error": ["non-ring atom", "Can't kekulize mol"],
    "Unclosed Ring": ["SMILES Parse Error: unclosed ring"],
    "Parentheses Error": ["SMILES Parse Error: extra close parentheses", "SMILES Parse Error: extra open parentheses"],
    "Valence Error": ["Explicit valence for atom", "Conflicting single bond"],
    "Syntax Error": ["SMILES Parse Error: syntax error", "SMILES Parse Error: Failed parsing SMILES"],
    "Bond Already Exists": ["SMILES Parse Error: duplicated ring closure", "SMILES Parse Error: ring closure"]
}

error_code = {
    "Aromaticity Error": "0",
    "Unclosed Ring": "1",
    "Parentheses Error": "2",
    "Valence Error": "3",
    "Syntax Error": "4",
    "Bond Already Exists": "5"
}

count = {
    "0": 0,
    "1": 0,
    "2": 0,
    "3": 0,
    "4": 0,
    "5": 0
}

vocab = {
    "0": '>',
    "1": '<',
    "2": '?',
    "3": ';',
    "4": ':',
    "5": '&'
}

def error_type_results():
    result = {}
    with open(infile, 'r') as f:
        lines = f.readlines()
        lines = [s for s in lines if s != '\n']

    with open(outfile, 'w+') as w:
        for index, line in enumerate(lines):
            if line.startswith('['):  # 如果以'['开头
                for error_category, error_messages in error_categories.items():
                    split_error_message = line.split('] ', 1)  
                    for error_message in error_messages:
                        if len(split_error_message) > 1 and split_error_message[1].startswith(error_message):
                            error_num = error_code[error_category]
                            count[error_num] += 1
                            i = index + 1
                            if lines[i] == '\n':
                                i += 1
                            while lines[i].startswith('['):
                                i += 1
                            source_smiles = lines[i].split('\t')[0].strip()
                            if result.get(source_smiles):
                                result[source_smiles][0] = result[source_smiles][0] + error_num
                            else:
                                result[source_smiles] = [error_num, source_smiles,'C']
        for k, v in result.items():
            err = ''
            for vo in v[0]:
                err += vocab[vo]
            w.write("C\t"+err+v[1]+"\tM\n")
            # error_type right wrong
            # print('successful write')


error_type_results()
print(count)

