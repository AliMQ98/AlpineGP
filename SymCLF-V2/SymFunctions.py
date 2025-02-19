# Define Functions related to individuals
from alpine.gp import util
import re
from sympy import sympify

def compile_individuals(toolbox, individuals_str_batch):
    """
    Compiles a batch of individuals (symbolic expressions) using a given toolbox.

    Parameters:
        toolbox (object): A DEAP (Distributed Evolutionary Algorithms in Python) toolbox containing the compile method.
        individuals_str_batch (list of str): A list of symbolic expressions in string format.

    Returns:
        list: A list of compiled symbolic expressions.
    """
    return [toolbox.compile(expr=ind) for ind in individuals_str_batch]

def check_trig_fn(ind):
    """
    Counts the occurrences of trigonometric functions (sin, cos) in a symbolic expression.

    Parameters:
        ind (str): A symbolic expression in string format.

    Returns:
        int: The count of trigonometric functions present in the expression.
    """
    return len(re.findall("cos", str(ind))) + len(re.findall("sin", str(ind)))


def check_nested_trig_fn(ind):
    """
    Detects nested trigonometric functions in a symbolic expression.

    Parameters:
        ind (str): A symbolic expression in string format.

    Returns:
        int: The count of nested trigonometric functions detected.
    """
    return util.detect_nested_trigonometric_functions(str(ind))


def get_features_batch(
    individuals_str_batch,
    individ_feature_extractors=[len, check_nested_trig_fn, check_trig_fn],
):
    
    """
    Extracts multiple features from a batch of symbolic expressions.

    Parameters:
        individuals_str_batch (list of str): A list of symbolic expressions.
        individ_feature_extractors (list of functions, optional): A list of functions to extract features.
            Defaults to [len, check_nested_trig_fn, check_trig_fn].

    Returns:
        tuple: Three lists containing:
            - individ_length (list): Length of each expression.
            - nested_trigs (list): Number of nested trigonometric functions in each expression.
            - num_trigs (list): Total number of trigonometric functions in each expression.
    """
    
    features_batch = [
        [fe(i) for i in individuals_str_batch] for fe in individ_feature_extractors
    ]

    individ_length = features_batch[0]
    nested_trigs = features_batch[1]
    num_trigs = features_batch[2]
    return individ_length, nested_trigs, num_trigs

def read_expression(file_path):
    """
    Reads a symbolic expression from the second line of a text file.

    Parameters:
        file_path (str): The path to the file.

    Returns:
        str or None: The expression (if found), otherwise None.
    """
    
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

    Parameters:
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
