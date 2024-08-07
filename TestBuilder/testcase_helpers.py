import os
from tree_sitter import Language, Parser
import faiss
from neo4j import GraphDatabase
import openai
from django.conf import settings
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.preprocessing import normalize
PYTHON_LANGUAGE = Language(os.path.join(settings.BASE_DIR, 'tree-sitter-python/libtree-sitter-python.so'), 'python')

parser = Parser()

parser.set_language(PYTHON_LANGUAGE)
def setup_openai(api_key, api_type, api_base, api_version):
    openai.api_key = api_key
    openai.api_type = api_type
    openai.api_base = api_base
    openai.api_version = api_version
    

class EmbeddingGenerator:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
    
    def get_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
        return embeddings.squeeze().numpy()

embedding_generator = EmbeddingGenerator()

class Neo4jGraph:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def create_function_node(self, function_name, file_name,code):
        with self.driver.session() as session:
            session.write_transaction(self._create_function_node, function_name, file_name,code)

    def create_import_node(self, import_name):
        with self.driver.session() as session:
            session.write_transaction(self._create_node, import_name)

    def create_import_relationship(self, function_name, import_name):
        with self.driver.session() as session:
            session.write_transaction(self._create_relationship, function_name, import_name)

    @staticmethod
    def _create_function_node(tx, function_name, file_name,code):
        tx.run("MERGE (n:Node {name: $name, file: $file, code: $code})", name=function_name, file=file_name, code=code)

    @staticmethod
    def _create_node(tx, name):
        tx.run("MERGE (n:Node {name: $name})", name=name)

    @staticmethod
    def _create_relationship(tx, function_name, import_name):
        tx.run("""
            MATCH (f:Node {name: $function_name})
            MATCH (i:Node {name: $import_name})
            MERGE (f)-[:DEPENDS_ON]->(i)
            """, function_name=function_name, import_name=import_name)
    
    def get_all_functions(self):
        with self.driver.session() as session:
            result = session.execute_read(self._query_all_functions)
            return result

    @staticmethod
    def _query_all_functions(tx):
        query = """
        MATCH (f:Node)
        WHERE f.name IS NOT NULL
        RETURN f.name AS function_name, f.file AS file_name, f.code as code
        """
        result = tx.run(query)
        return [{'function_name': record['function_name'], 'file_name': record['file_name'],'code': record['code']} for record in result]

    def get_related_data(self, function_name):
        with self.driver.session() as session:
            result = session.execute_read(self._query_related_data, function_name)
            return result
        
    def create_test_case_prompt(self, function_name, file_name, related_data,code,context):
        imports = ', '.join(related_data['imports'])
        related_functions = ', '.join(related_data['related_functions'])
        context = '\n '.join(context)
        prompt = f"""
        Generate unit tests for the following function:

        File: {file_name}
        Function: {code}
        Context: {context}
        Instructions:
        1. The function has the following imports: {imports}
        2. It depends on the following related functions: {related_functions}
        3. Analyze the provided related functions and imports to determine the module or app that each model or function belongs to.
        4. If the function uses a model, identify its module name or path from the given imports.
        5. Do not make any assumptions or import models or functions from unspecified modules. Use the context provided by the imports and related functions to proceed.

        Based on this information, generate comprehensive unit tests to ensure the function works as expected.
        Make sure the data generated output just have the unit test cases not other content
        Ensure you always specify the correct module or app name; avoid using placeholder names like `your_module` or `my_app`. As everything is provided there in context.
        Write comprehensive unit tests in python to ensure the function works as expected, considering its dependencies and interactions.
        """
        messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant specialized in writing Python unit tests."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
        print("prompt:: ",prompt)
        return messages
        
    def extract_function_code(self, code, function_name):
        tree = parser.parse(bytes(code, 'utf8'))
        def find_function_node(node):
            if node.type == 'function_definition':
                name_node = node.child_by_field_name('name')
                if name_node and name_node.text.decode('utf8') == function_name:
                    return node
            for child in node.children:
                result = find_function_node(child)
                if result:
                    return result
            return None

        function_node = find_function_node(tree.root_node)
        if not function_node:
            return None

        function_start = function_node.start_byte
        function_end = function_node.end_byte
        function_code = code[function_start:function_end]

        return function_code

    def generate_test_cases(self, openai_type, function_name, file_name,code,similar_functions):
        related_data = self.get_related_data(function_name)
        context_set = set()
        for fpath, fname in similar_functions:
            if fname not in context_set:
                context_set.add(f"Function: {fname}\nCode:\n{self.extract_function_code(open(fpath, 'r').read(), fname)}")
        if related_data:
            messages = self.create_test_case_prompt(function_name, file_name, related_data[0],code,context_set)
            if openai_type == "azure":
                response = openai.ChatCompletion.create(
                    engine="gpt35turbo",  # or any other available model
                    messages=messages,
                    temperature=0.5
                )
            else:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",  # or any other available model
                    messages=messages,
                    temperature=0.5
                )
            return response['choices'][0]['message']['content'].strip().replace("'''","")
        return None

    @staticmethod
    def _query_related_data(tx, function_name):
        query = """
        MATCH (f:Node {name: $function_name})-[:DEPENDS_ON]->(i:Node)
        RETURN f.name AS function, COLLECT(DISTINCT i.name) AS imports
        """
        result = tx.run(query, function_name=function_name)
        data = []
        for record in result:
            imports = record['imports']
            data.append({
                'function': record['function'],
                'imports': imports
            })
        
        # Also fetch related functions
        related_functions_query = """
        MATCH (f:Node {name: $function_name})-[:DEPENDS_ON]->(related:Node)
        WHERE related.name <> $function_name
        RETURN related.name AS related_function
        """
        related_result = tx.run(related_functions_query, function_name=function_name)
        related_functions = [record['related_function'] for record in related_result]
        
        for item in data:
            item['related_functions'] = related_functions
        
        return data


# Function to parse files and extract functions and imports
def parse_file(filepath):
    print("filepath((()))",filepath,flush=True)
    with open(filepath, 'r') as file:
        code = file.read()
    print("code", code,flush=True)
    tree = parser.parse(bytes(code, 'utf8'))
    return tree, code

def extract_functions_and_imports(tree, code):
    root_node = tree.root_node
    functions = []
    imports = []

    def traverse_node(node):
        if node.type == 'function_definition':
            function_name = code[node.child_by_field_name("name").start_byte:node.child_by_field_name("name").end_byte]
            functions.append((function_name, node.start_byte, node.end_byte,code[node.start_byte:node.end_byte]))
        elif node.type in ['import_statement', 'import_from_statement']:
            import_statement = code[node.start_byte:node.end_byte]
            imports.append(import_statement)
        for child in node.children:
            traverse_node(child)

    traverse_node(root_node)
    return functions, imports



def get_function_embedding(function_code):
    embedding = embedding_generator.get_embedding(function_code)
    return np.array(embedding, dtype=np.float32)


def walk_project_directory(directory, graph):
    apps = [app for app in os.listdir(directory) if os.path.isdir(os.path.join(directory, app))]
    function_embeddings = []
    function_ids = []
    for app in apps:
        app_path = os.path.join(directory, app)
        for root, _, files in os.walk(app_path):
            for file in files:
                if file.endswith(".py") and file not in ["__init__.py", "tests.py","settings.py","urls.py","wsgi.py"]:
                    filepath = os.path.join(root, file)
                    tree, code = parse_file(filepath)
                    functions, imports = extract_functions_and_imports(tree, code)

                    for func_name, _, _, code in functions:
                        embedding = get_function_embedding(code)
                        function_embeddings.append(embedding)
                        function_ids.append((filepath, func_name))
                        graph.create_function_node(func_name.strip(), filepath,code)

                    for imp in imports:
                        graph.create_import_node(imp.strip())

                    for func_name, _, _, _ in functions:
                        for imp in imports:
                            graph.create_import_relationship(func_name.strip(), imp.strip())
    
    matrix = np.vstack(function_embeddings)
    index = faiss.IndexFlatL2(matrix.shape[1])
    index.add(matrix)
    return index, function_ids

def write_tests_to_file(data):
    for app_name, test_cases in data.items():
        tests_file_path = os.path.join(app_name, 'tests.py')
        
        exclude_paths = ['migrations', '__init__.py']
        
        all_imports = set()
        for file_path in test_cases.keys():
            if any(exclude in file_path for exclude in exclude_paths):
                continue
            
            module_name = file_path.replace('/', '.').replace('.py', '')
            all_imports.add(f"from {module_name} import *")

        with open(tests_file_path, 'w+') as tests_file:
            for _, test_case in test_cases.items():
                if test_case:
                    test_case = test_case.replace("python", "")  
                    test_lines = test_case.split('\n')
                    for line in test_lines:
                        line = line.replace("```", "")
                        if line.strip(): 
                            tests_file.write(f"{line}\n")
                    tests_file.write("\n\n")

        print(f"Tests written to {tests_file_path}")




def find_similar_functions(index, function_ids, embedding, top_k=5):
    embedding = normalize(np.array([embedding], dtype=np.float32), axis=1)
    _, indices = index.search(embedding, top_k)
    similar_functions = [function_ids[idx] for idx in indices[0]]
    return similar_functions

def create_testcases(openai_type, project_path,neo4j_url, username, password):
    graph = Neo4jGraph(neo4j_url, username, password)
    try:
        # Walk through the Django project directory
        index, function_ids = walk_project_directory(project_path, graph)
        
        all_functions = graph.get_all_functions()
        test_cases_dict = {}
        for function in all_functions:
            function_name = function['function_name']
            if function.get("file_name"):
                file_name = function['file_name']
                print(f"Processing function: {function_name} in file: {file_name}")
                embedding = get_function_embedding(function["code"])

                # Find similar functions for better context
                similar_functions = find_similar_functions(index, function_ids, embedding,2)
                test_cases = graph.generate_test_cases(openai_type, function_name, file_name,function["code"],similar_functions)
                directory, _, _ = file_name.rpartition('/')
                if test_cases_dict.get(directory):
                    test_cases_dict[directory][function_name] = test_cases
                else:
                    test_cases_dict[directory] = {function_name: test_cases}
                print(f"Test cases for function {function_name}:")
                print(test_cases)
                print('-' * 40)
        write_tests_to_file(test_cases_dict)
    finally:
        graph.close()
