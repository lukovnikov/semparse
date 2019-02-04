from semparse.trees.tree import Node
from semparse.trees.parse import parse_lambda_depth_first_parentheses
import re


def print_lambda_depth_first_stop_token(node:Node):
    ret = node.name
    if len(node.children) > 0:
        ret += " " \
            +  " ".join([print_lambda_depth_first_stop_token(child) for child in node.children]) \
            + " <END>"
    return ret


def print_lambda_depth_first_parentheses(node):
    ret = node.name
    if len(node.children) > 0:
        ret = "( " \
            + "{} ".format(ret) \
            + " ".join([print_lambda_depth_first_parentheses(child) for child in node.children]) \
            + " )"
    return ret

def print_lambda_breadth_first(node):
    ret = ""
    queue = [node]
    while len(queue) > 0:
        head = queue.pop(0)
        ret += head.name
        if len(head.children) > 0:
               ret += " " + " ".join(["$$" for _ in head.children])
        ret += " <END> "
        queue += list(head.children)
    return ret


if __name__ == '__main__':
    s = "( lambda $0 e ( exists $1 ( and ( flight $0 ) ( flight $1 ) ( during_day $1 late_evening:pd ) ( during_day $0 early:pd ) ( during_day $0 morning:pd ) ( from $0 ci0 ) ( to $0 ci1 ) ( to $1 ci0 ) ( from $1 ci1 ) ) ) )"
    n = parse_lambda_depth_first_parentheses(s)
    print(n.pptree())
    print(print_lambda_depth_first_stop_token(n))
    print(print_lambda_depth_first_parentheses(n))
    print(print_lambda_breadth_first(n))
    assert(print_lambda_depth_first_parentheses(n) == s)