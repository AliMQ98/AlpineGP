# Define Functions related to individuals
from alpine.gp import util
import re
from sympy import sympify

def compile_individuals(toolbox, individuals_str_batch):
    return [toolbox.compile(expr=ind) for ind in individuals_str_batch]

def check_trig_fn(ind):
    return len(re.findall("cos", str(ind))) + len(re.findall("sin", str(ind)))


def check_nested_trig_fn(ind):
    return util.detect_nested_trigonometric_functions(str(ind))


def get_features_batch(
    individuals_str_batch,
    individ_feature_extractors=[len, check_nested_trig_fn, check_trig_fn],
):
    features_batch = [
        [fe(i) for i in individuals_str_batch] for fe in individ_feature_extractors
    ]

    individ_length = features_batch[0]
    nested_trigs = features_batch[1]
    num_trigs = features_batch[2]
    return individ_length, nested_trigs, num_trigs

def read_expression(file_path):
    try:
        with open(file_path, "r") as file:
            lines = file.readlines()  # Read all lines into a list
            if len(lines) > 1:
                expression = lines[1].strip()  # Second line contains the expression
                return expression
            else:
                print("File does not contain enough lines.")
                return None
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
        return None

def DeapSimplifier(ind):
    """
    Processes and simplifies a symbolic expression using custom operators.

    Args:
        ind (str): A string representing the symbolic expression.

    Returns:
        sympy expression: A simplified version of the input expression.
    """
    locals = {
        'sub': lambda x, y: x - y,
        'div': lambda x, y: x / y,
        'mul': lambda x, y: x * y,
        'add': lambda x, y: x + y,
        'neg': lambda x: -x,
        'pow': lambda x, y: x ** y,
    }

    print(f'Original Expression: {ind}')
    expr = sympify(str(ind), locals=locals)
    print(f'Simplified Expression: {expr}')
    return expr
