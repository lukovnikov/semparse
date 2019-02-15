from semparse.trees.tree import Node
import numpy as np
import re


def parse_lambda_depth_first_parentheses(s):
    """
    Example input:
        ( lambda $0 e ( exists $1 ( and ( flight $0 ) ( flight $1 ) ( during_day $1 late_evening:pd ) ( during_day $0 early:pd ) ( during_day $0 morning:pd ) ( from $0 ci0 ) ( to $0 ci1 ) ( to $1 ci0 ) ( from $1 ci1 ) ) ) )
    :param s:
    :return:
    """
    try:
        ret = []
        splits = s.split()[1:-1]
        stack = []
        stack.append(ret)
        for i in range(len(splits)):
            if splits[i] == "(":
                stack[-1].append([])
                stack.append(stack[-1][-1])
            elif splits[i] == ")":
                stack.pop(-1)
            else:
                stack[-1].append(splits[i])
        ret = nested_lists_to_tree(ret)
    except Exception as e:
        ret = Node("badtree - " + str(np.random.randint(0, 100000)))
    return ret


def parse_prolog(s):
    """ Example input:
        _answer ( A , _largest ( A , ( _city ( A ) , _loc ( A , B ) , _const ( B , _stateid ( kansas ) ) ) ) )
    """
    try:
        ret = []
        splits = s.split()
        stack = []
        stack.append(ret)
        for i in range(len(splits)):
            if splits[i] == "(":
                parentname = stack[-1].pop(-1)
                stack[-1].append([parentname])
                stack.append(stack[-1][-1])
            elif splits[i] == ")":
                if stack[-1][0] == ",":
                    stack[-1][0] = "and"
                j = 0
                while j < len(stack[-1]):
                    if stack[-1][j] == ",":
                        stack[-1].pop(j)
                        j -= 1
                    j += 1
                stack.pop(-1)
            else:
                stack[-1].append(splits[i])
        ret = ret[0]
        ret = nested_lists_to_tree(ret)
    except Exception as e:
        ret = Node("badtree - " + str(np.random.randint(0, 100000)))
    return ret


def nested_lists_to_tree(nls):
    topnode = _rec_build_tree_lambda(nls)

    def remove_order_lambda(a):
        if a.name == "and":
            for achild in a.children:
                achild.order = None

    topnode.apply(remove_order_lambda)
    return topnode


def _rec_build_tree_lambda(x):
    if isinstance(x, list) and len(x) > 1:
        name = x[0]
        children = [_rec_build_tree_lambda(child) for child in x[1:]]
        i = 0
        for child in children:
            child.order = i
            i += 1
        ret = Node(name, children=children)
    else:
        if isinstance(x, list):
            x = x[0]
        ret = Node(x)
    return ret


def parse_lambda_breadth_first(s):
    """
    Example input:
        lambda $$ $$ $$ <END> $0 <END> e <END> exists $$ $$ <END> $1 <END> and $$ $$ $$ $$ $$ $$ $$ $$ $$ <END> flight $$ <END> $0 <END> flight $$ <END> $1 <END> during_day $$ $$ <END> $1 <END> late_evening:pd <END> during_day $$ $$ <END> $0 <END> early:pd <END> during_day $$ $$ <END> $0 <END> morning:pd <END> from $$ $$ <END> $0 <END> ci0 <END> to $$ $$ <END> $0 <END> ci1 <END> to $$ $$ <END> $1 <END> ci0 <END> from $$ $$ <END> $1 <END> ci1 <END>
    :param s:
    :return:
    """
    ret = []
    queue = [ret]
    splits = s.split()
    for i in range(len(splits)):
        if splits[i] == "$$":
            n = []
            queue[0].append(n)
            queue.append(n)
        elif splits[i] == "<END>":  # take next item from queue
            queue.pop(0)
        else:
            queue[0].append(splits[i])

    ret = nested_lists_to_tree(ret)
    return ret


if __name__ == '__main__':
    r = parse_lambda_depth_first_parentheses("( lambda $0 e ( exists $1 ( and ( flight $0 ) ( flight $1 ) ( during_day $1 late_evening:pd ) ( during_day $0 early:pd ) ( during_day $0 morning:pd ) ( from $0 ci0 ) ( to $0 ci1 ) ( to $1 ci0 ) ( from $1 ci1 ) ) ) )")
    print(r)
    print(r.pptree())

    r = parse_prolog("_answer ( A , _largest ( A , ( _city ( A ) , _loc ( A , B ) , _const ( B , _stateid ( kansas ) ) ) ) )")
    print(r.pptree())

    # rbf = "lambda $$ $$ $$ <END> $0 <END> e <END> exists $$ $$ <END> $1 <END> and $$ $$ $$ $$ $$ $$ $$ $$ $$ <END> flight $$ <END> flight $$ <END> during_day $$ $$ <END> during_day $$ $$ <END> during_day $$ $$ <END> from $$ $$ <END> to $$ $$ <END> to $$ $$ <END> from $$ $$ <END> $0 <END> $1 <END> $1 <END> late_evening:pd <END> $0 <END> early:pd <END> $0 <END> morning:pd <END> $0 <END> ci0 <END> $0 <END> ci1 <END> $1 <END> ci0 <END> $1 <END> ci1 <END>"
    # r2 = parse_lambda_breadth_first(rbf)
    # print(r2.pptree())
    # assert(r2 == r)
    # print("SAME")