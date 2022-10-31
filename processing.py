import enum
import os
import json
import argparse
import time
from tqdm import tqdm
import multiprocessing
from multiprocessing import Manager, Value
# from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import pandas as pd
from datasets import load_dataset
from tree_sitter import Parser, Language
from docstring_parser.common import ParseError

from utils.languages_function import export_jsonl
from utils.parser.go_parser import GoParser
from utils.parser.ruby_parser import RubyParser
from utils.parser.php_parser import PhpParser
from utils.parser.java_parser import JavaParser
from utils.parser.javascript_parser import JavascriptParser
from utils.parser.python_parser import PythonParser
from utils.tree_utils import import_language_parser, reformat_function_data, reformat_line_data, reformat_class_data


def args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--data_file', type=str, default='./data/raw/python_data.jsonl')
    parser.add_argument('--language', type=str, default='Python')
    parser.add_argument('--save_path', type=str, default='./data/python/')
    parser.add_argument('--data_path', type=str, default='./cache')
    parser.add_argument('-s', '--split', type=int, default=40)

    return parser.parse_args()


def _processing(dataset, indexs, ast, lang_parser, idx=None, is_file=None):
    for idx in tqdm(indexs, desc=f'Thread {idx}'):
    # for data in tqdm(dataset, desc=f'Thread {idx}'):
        data = dataset[idx]
        if is_file:
            data = json.loads(data)
        
        try:
            processed_data = {
                "repo": data["repo_name"],
                "path": data["path"],
                "language": data["language"],
                "license": data["license"],
            }
        except:
            raise ValueError('Mismatch key')
        
        raw_code = data["code"]
        tree = ast.parse(bytes(raw_code, "utf8"))
        
        # try:
        fn_metadata = list(lang_parser.get_function_definitions(tree, raw_code))
        class_metadata = list(lang_parser.get_class_definitions(tree, raw_code))
        line_metadata = list(lang_parser.get_line_definitions(tree, raw_code))
        
        fn_data, class_data, line_data = [], [], []
        if len(fn_metadata) > 0:
            fn_data = reformat_function_data(processed_data, fn_metadata)
        
        if len(class_metadata) > 0:
            class_data = reformat_class_data(processed_data, class_metadata)
        
        if len(line_metadata) > 0:
            line_data = reformat_line_data(processed_data, line_metadata)

        # We only take function which has docstring (block_comment) and
        # their docstring is larger than 3 words and smaller than 256 words
        for item in fn_data:
            if item['docstring']['block_comment'] == None:
                
                continue
            if len(item['docstring_tokens']) <= 3 or len(item['docstring_tokens']) >= 256:
                continue
            
            yield item
        
        for item in class_data:
            if len(item['docstring_tokens']) <= 3 or len(item['docstring_tokens']) >= 256:
                continue
            
            yield item
            

        for item in line_data:
            if len(item['comment']) <= 3 or len(item['comment']) >= 256:
                continue
            
            yield item
            
        # except Exception: # (ParseError, AttributeError, TypeError, UnboundLocalError):
        #     # with open(os.path.join(os.path.dirname(save_path), f'{id}_fail.jsonl'), 'a') as file:
        #     #     json.dump(data, file)
        #     #     file.write('
        # ')
        #     pass
        

def processing(dataset, index, language, save_path, idx=None, is_file=None):
    # setup language parser
    language = str(language).lower()
    if language == "c++": language = "cpp"
    if language == "c#": language = "c_sharp"
    
    ast_parser = Parser()
    tree_language = Language('./languages/my-languages.so', language)
    ast_parser.set_language(tree_language)
    
    if language == 'python':
        language_parser = PythonParser()

    elif language == 'java':
        language_parser = JavaParser()
    
    elif language == 'javascript':
        language_parser = JavascriptParser()
        
    elif language == 'go':
        language_parser = GoParser()
        
    elif language == 'ruby':
        language_parser = RubyParser()

    elif language == 'php':
        language_parser = PhpParser()
        
    else:
        raise ValueError(f'Language {language} not supported')
    # list_function = list(_processing(dataset, index, ast_parser, language_parser, idx))
    list_data = _processing(dataset, index, ast_parser, language_parser, idx, is_file)
    
    n_fn, n_class, n_line = 0, 0, 0
    fn_file = open(os.path.join(save_path, f'batch_{idx}_function_data.jsonl'), "a")
    class_file = open(os.path.join(save_path, f'batch_{idx}_class_data.jsonl'), "a")
    line_file = open(os.path.join(save_path, f'batch_{idx}_line_data.jsonl'), "a")
    
    for function in list_data:
        if 'func_name' in function.keys():
            n_fn += 1
            json.dump(function, fn_file, ensure_ascii=False)
            fn_file.write('\n')
            
        elif 'class_name' in function.keys():
            n_class += 1
            json.dump(function, class_file, ensure_ascii=False)
            class_file.write('\n')
        
        else:
            n_line += 1
            json.dump(function, line_file, ensure_ascii=False)
            line_file.write('\n')
        
    fn_file.close()
    class_file.close()
    line_file.close()
            
    return n_fn, n_class, n_line


def start_executor(dataset, language, save_path, split, is_file):
    """
    Multi-processing on CPUs
    
    :param dataset: huggingface dataset or list of json object
    :param language: language
    :param save_path: path to discrete save
    :param n: split dataset into n file. 
    """
    # Start multiprocessing
    n_worker = multiprocessing.cpu_count()
    print(f'Using {n_worker} cores.')
    
    dataset_size = len(dataset)
    index_list = range(dataset_size)
    chunk_size = dataset_size//split
    
    jobs_list = [index_list[x:x+chunk_size] for x in range(0, dataset_size, chunk_size)]  # n set
    # jobs_list = [dataset[x:x+chunk_size] for x in range(0, dataset_size, chunk_size)]  # n set
    
    # futures = []
    # args = []
    for idx, job_index in enumerate(jobs_list):
        # args.append([dataset, job_index, language, save_path, idx, is_file])
        # for test 1 process
        processing(dataset, job_index, language, save_path, is_file=is_file)

    # executor = multiprocessing.Pool(n_worker)
    # result = list(executor.starmap(processing, args))
                
    # total_fn, total_class, total_line = 0, 0, 0
    # for res in result:
    #     total_fn += res[0]
    #     total_class += res[1]
    #     total_line += res[2]
    
    # print(f'
    # ========================
    # Total sample | Function: {total_fn} | Class: {total_class} | Line: {total_line}') 

if __name__ == '__main__':
    opt = args()
    split, language, save_path, data_path = opt.split, opt.language, opt.save_path, opt.data_path
    
    # debug
    # data_path = '/media/Z/dungnm31/small_100k_dataset/python/raw/small_data.jsonl'
    # save_path = '/media/Z/dungnm31/small_100k_dataset/python/test'
    
    is_file = False
    try:
        # load .jsonl file
        with open(data_path, 'r') as json_file:
            dataset = list(json_file)
        is_file = True

    except (FileNotFoundError, IsADirectoryError):
        # load huggingface cache file 
        dataset = load_dataset("codeparrot/github-code", languages=[language], split='train', cache_dir=data_path)
        
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    

    # Start processing
    start = time.perf_counter()
    start_executor(dataset, language, save_path, split, is_file)
    
    # executor.shutdown()
    finish = time.perf_counter()
    print(f'Finished in {(finish - start):.2f} seconds')
