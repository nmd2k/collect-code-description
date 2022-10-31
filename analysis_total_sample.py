import os
import json
from tqdm import tqdm
from datasets import load_dataset
from tree_sitter import Parser
from utils.tree_utils import import_language_parser
from utils.parser.language_parser import traverse_type

tree_dict = import_language_parser()
# cache_path = '/media/Z/dungnm31/small_500k_dataset/javascript/small_data.jsonl'
cache_path = '/media/Z/dungnm31/dataset/java/cache'

language = 'Java'
dataset = load_dataset("codeparrot/github-code", languages=[language], cache_dir=cache_path, split='train')

language = language.lower()
parser = Parser()
parser.set_language(tree_dict[language])

count_function = 0
fail = 0

index = range(500000)
# with open(cache_path, 'r') as json_file:
#     dataset = list(json_file)

for item in tqdm(index):
    # data = json.loads(item)
    data = dataset[item]
    code = data['code']
    
    tree = parser.parse(bytes(code, 'utf8'))
    root_node = tree.root_node
    
    function = []
    try:
        if language == 'python':
            traverse_type(root_node, function, ['function_definition', 'decorated_definition'])

        elif language == 'java':
            traverse_type(root_node, function, 'method_declaration')
            
        elif language == 'javascript':
            traverse_type(root_node, function, 'function')
    
    except Exception:
        fail += 1
        continue
        
    count_function += len(function)

print('Total extractable function', count_function)
print('Fail', fail)
        
    
    


