from collections import OrderedDict
import random
import numpy as np
from unidecode import unidecode


class NodeType(object):
    def __init__(self, num_children=None, min_children=None, max_children=None, **kw):
        super(NodeType, self).__init__(**kw)
        assert((num_children is not None) or (min_children is not None))        # one of them must be specified
        assert(not ((num_children is not None) and (min_children is not None))) # not both of them should be specified
        assert(not ((num_children is not None) and (max_children is not None))) # not both of them should be specified
        if num_children is not None:
            min_children = num_children
            max_children = num_children
        if max_children is None:
            max_children = np.infty
        self.min_children = min_children
        self.max_children = max_children

    def check(self, node):
        assert(len(node.children) >= self.min_children
               and len(node.children) <= self.max_children)


class Node(object):
    """
    Node has children, some of which may be ordered.
    """
    def __init__(self, name, order=None, children=tuple(), nodetype=None, **kw):
        super(Node, self).__init__(**kw)
        self.name = name
        self.order = order
        self.nodetype = nodetype    # keeps specs of node (number of children etc)
        self.children = tuple(sorted(children, key=lambda x: x.order if x.order is not None else np.infty))
        if self.nodetype is not None:
            self.nodetype.check(self)

    @property
    def is_leaf(self):
        return len(self.children) == 0

    def equals(self, other):
        if other is None:
            return False
        ret = self._eq_rec(other)
        return ret

    def __eq__(self, other):
        return self.equals(other)

    def apply(self, f):
        f(self)
        for child in self.children:
            child.apply(f)

    def _eq_rec(self, other, _self_pos_in_list=None, _other_pos_in_list=None):
        if isinstance(other, list):
            print("other is list")
            return False
        same = self.name == other.name
        # same &= self.order == other.order
        same &= _self_pos_in_list == _other_pos_in_list
        ownchildren = self.children + tuple()
        otherchildren = other.children + tuple()
        j = 0
        while j < len(ownchildren):
            child = ownchildren[j]
            order_matters = False
            if child.order is not None:
                order_matters = True
            found = False
            i = 0
            while i < len(otherchildren):
                otherchild = otherchildren[i]
                if otherchild.order is not None:
                    order_matters = True
                equality = child._eq_rec(otherchild, _self_pos_in_list=j, _other_pos_in_list=i) \
                    if order_matters else child._eq_rec(otherchild)
                if equality:
                    found = True
                    break
                i += 1
            if found:
                otherchildren = otherchildren[:i] + otherchildren[i+1:]
                ownchildren = ownchildren[:j] + ownchildren[j+1:]
            else:
                j += 1
            same &= found
        same &= len(otherchildren) == 0 and len(ownchildren) == 0   # all children must be matched
        return same

    def symbol(self, with_annotation=True, with_order=True):
        ret = self.name
        if with_order:
            ret += "{}{}".format(":", self.order) if self.order is not None else ""
        if with_annotation:
            pass
            #if self.is_leaf and not self.name in [self.none_symbol, self.root_symbol]:
            #    ret += self.suffix_sep + self.leaf_suffix
        return ret

    def pptree(self, _rec_arg=None, _top_rec=True, _remove_order=False):
        direction = "root" if _top_rec else _rec_arg
        children = list(self.children)
        if len(self.children) > 0:
            def print_children(_children, _direction):
                _lines = []
                _dirs = ["up"] + ["middle"] * (len(_children) - 1) if _direction == "up" \
                    else ["middle"] * (len(_children) - 1) + ["down"]
                for elem, _dir in zip(_children, _dirs):
                    elemlines = elem.pptree(_rec_arg=_dir, _top_rec=False, _remove_order=_remove_order)
                    _lines += elemlines
                return _lines

            parent = self.symbol(with_annotation=False, with_order=not _remove_order)
            if isinstance(parent, str):
                parent = unidecode(parent)
            up_children, down_children = children[:len(children)//2], children[len(children)//2:]
            up_lines = print_children(up_children, "up")
            down_lines = print_children(down_children, "down")
            uplineprefix = "│" if direction == "middle" or direction == "down" else "" if direction == "root" else " "
            lines = [uplineprefix + " " * len(parent) + up_line for up_line in up_lines]
            parentprefix = "" if direction == "root" else '┌' if direction == "up" else '└' if direction == "down" else '├' if direction == "middle" else " "
            lines.append(parentprefix + parent + '┤')
            downlineprefix = "│" if direction == "middle" or direction == "up" else "" if direction == "root" else " "
            lines += [downlineprefix + " " * len(parent) + down_line for down_line in down_lines]
        else:
            connector = '┌' if direction == "up" else '└' if direction == "down" else '├' if direction == "middle" else ""
            s = self.symbol(with_annotation=False, with_order=not _remove_order)
            if isinstance(s, str):
                s = unidecode(s)
            lines = [connector + s]
        if not _top_rec:
            return lines
        ret = "\n".join(lines)
        return ret

    def __repr__(self):
        return self.pptree()


# DON'T USE BELOW
class ONode(object):
    """ !!! If mixed order children, children order is ordered children first, then unordered ones"""
    leaf_suffix = "NC"
    last_suffix = "LS"
    none_symbol = "<NONE>"
    root_symbol = "<ROOT>"

    suffix_sep = "*"
    label_sep = "/"
    order_sep = "#"

    def __init__(self, name, label=None, order=None, children=tuple(), **kw):
        super(ONode, self).__init__(**kw)
        self.name = name    # name must be unique in a tree
        self._label = label
        self._order = order
        self.children = tuple(sorted(children, key=lambda x: x.order if x.order is not None else np.infty))
        self._ordered_children = len(self.children) > 0 and self.children[0].order is not None

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, value):
        self._label = value

    @property
    def order(self):
        return self._order

    @order.setter
    def order(self, value):
        self._order = value

    @classmethod
    def parse_df(cls, inp, _toprec=True):
        tokens = inp
        if _toprec:
            tokens = tokens.replace("  ", " ").strip().split()

        otokens = tokens + []

        siblings = []

        while True:
            head, tokens = tokens[0], tokens[1:]
            xsplits = head.split(cls.suffix_sep)
            x, isleaf, islast = xsplits[0], cls.leaf_suffix in xsplits, cls.last_suffix in xsplits
            headname, headlabel = x, None
            if len(x.split(cls.label_sep)) == 2:
                headname, headlabel = x.split(cls.label_sep)
            headname, headorder = headname, None
            if len(headname.split(cls.order_sep)) == 2:
                headname, headorder = headname.split(cls.order_sep)

            children = tuple()
            if not isleaf:
                children, tokens = cls.parse_df(tokens, _toprec=False)

            newnode = ONode(headname, label=headlabel, order=headorder, children=children)
            siblings.append(newnode)

            if islast:
                if _toprec:
                    assert(len(siblings) == 1)
                    return siblings[0]
                else:
                    return siblings, tokens

    @classmethod
    def parse(cls, inp, _rec_arg=None, _toprec=True, _ret_remainder=False):
        """ breadth-first parse """
        tokens = inp
        if _toprec:
            tokens = tokens.replace("  ", " ").strip().split()

        otokens = tokens + []

        parents = _rec_arg + [] if _rec_arg is not None else None
        level = []
        siblings = []

        while True:
            head, tokens = tokens[0], tokens[1:]
            xsplits = head.split(cls.suffix_sep)
            isleaf, islast = cls.leaf_suffix in xsplits, cls.last_suffix in xsplits
            isleaf, islast = isleaf or head == cls.none_symbol, islast or head == cls.none_symbol
            x = xsplits[0]
            headname, headlabel = x, None
            if len(x.split(cls.label_sep)) == 2:
                headname, headlabel = x.split(cls.label_sep)
            headname, headorder = headname, None
            if len(headname.split(cls.order_sep)) == 2:
                headname, headorder = headname.split(cls.order_sep)

            newnode = ONode(headname, label=headlabel, order=headorder)
            if not isleaf:
                level.append(newnode)
            siblings.append(newnode)

            if islast:
                if _toprec:     # siblings are roots <- no parents
                    break
                else:
                    parents[0].children = tuple(siblings)
                    siblings = []
                    del parents[0]
                    if len(parents) == 0:
                        break

        remainder = tokens

        if len(tokens) > 0:
            if len(level) > 0:
                remainder = cls.parse(tokens, _rec_arg=level, _toprec=False)
        else:
            assert (len(level)) == 0

        if _toprec:
            if len(siblings) == 1:
                ret = siblings[0]
                ret.delete_nones()
                if _ret_remainder:
                    return ret, (otokens, remainder)
                else:
                    return ret
            else:
                raise Exception("siblings is a list")
                return siblings
        else:
            return remainder

    def delete_nodes(self, *nodes):
        i = 0
        newchildren = []
        while i < len(self.children):
            child = self.children[i]
            for node in nodes:
                if child.name == node.name and child.label == node.label:
                    pass
                else:
                    newchildren.append(child)
            i += 1
        self.children = tuple(newchildren)
        for child in self.children:
            child.delete_nodes(*nodes)

    def delete_nones(self):
        self.delete_nodes(ONode(ONode.none_symbol))

    @property
    def is_leaf(self):
        return len(self.children) == 0

    @property
    def num_children(self):
        return len(self.children)

    @property
    def size(self):
        return 1 + sum([child.size for child in self.children])

    def __str__(self):  return self.pp()
    def __repr__(self): return self.symbol(with_label=True, with_annotation=True, with_order=True)

    def symbol(self, with_label=True, with_annotation=True, with_order=True):
        ret = self.name
        if with_order:
            ret += "{}{}".format(self.order_sep, self.order) if self.order is not None else ""
        if with_label:
            ret += self.label_sep + self.label if self.label is not None else ""
        if with_annotation:
            if self.is_leaf and not self.name in [self.none_symbol, self.root_symbol]:
                ret += self.suffix_sep + self.leaf_suffix
        return ret

    def pptree(self, arbitrary=False, _rec_arg=False, _top_rec=True):
        return self.pp(mode="tree", arbitrary=arbitrary, _rec_arg=_rec_arg, _top_rec=_top_rec)

    def ppdf(self, mode="par", arbitrary=False):
        mode = "dfpar" if mode == "par" else "dfann"
        return self.pp(mode=mode, arbitrary=arbitrary)

    def pp(self, mode="ann", arbitrary=False, _rec_arg=None, _top_rec=True, _remove_order=False):
        assert(mode in "ann tree dfpar dfann".split())
        children = list(self.children)

        if arbitrary is True:
            # randomly shuffle children while keeping children with order in positions they were in
            fillthis = [child if child._order is not None else None for child in children]
            if None in fillthis:
                pass
            children_without_order = [child for child in children if child._order is None]
            random.shuffle(children_without_order)
            for i in range(len(fillthis)):
                if fillthis[i] is None:
                    fillthis[i] = children_without_order[0]
                    children_without_order = children_without_order[1:]
            children = fillthis
        elif arbitrary in ("alphabetical", "psychical", "omegal"):    # psychical and omegal are both reverse alphabetical
            # randomly shuffle children while keeping children with order in positions they were in
            fillthis = [child if child._order is not None else None for child in children]
            if None in fillthis:
                pass
            children_without_order = [child for child in children if child._order is None]
            sortreverse = True if arbitrary in ("psychical", "omegal") else False
            children_without_order = sorted(children_without_order, key=lambda x: x.name, reverse=sortreverse)
            # random.shuffle(children_without_order)
            for i in range(len(fillthis)):
                if fillthis[i] is None:
                    fillthis[i] = children_without_order[0]
                    children_without_order = children_without_order[1:]
            children = fillthis
        elif arbitrary in ("heavy", "light"):
            # randomly shuffle children while keeping children with order in positions they were in
            fillthis = [child if child._order is not None else None for child in children]
            children_without_order = [child for child in children if child._order is None]
            sortreverse = True if arbitrary == "heavy" else False
            children_without_order = sorted(children_without_order, key=lambda x: x.size, reverse=sortreverse)
            if None in fillthis:
                pass
            # random.shuffle(children_without_order)
            for i in range(len(fillthis)):
                if fillthis[i] is None:
                    fillthis[i] = children_without_order[0]
                    children_without_order = children_without_order[1:]
            children = fillthis

        # children = sorted(children, key=lambda x: x.order if x.order is not None else np.infty)
        if mode == "dfpar":     # depth-first with parentheses
            children = [child.pp(mode=mode, arbitrary=arbitrary, _remove_order=_remove_order) for child in children]
            ret = self.symbol(with_label=True, with_annotation=False, with_order=not _remove_order) \
                  + ("" if len(children) == 0 else " ( {} )".format(" , ".join(children)))
        if mode == "dfann":
            _is_last = True if _rec_arg is None else _rec_arg
            children = [child.pp(mode=mode, arbitrary=arbitrary, _rec_arg=_is_last_child, _remove_order=_remove_order)
                        for child, _is_last_child
                        in zip(children, [False] * (len(children)-1) + [True])]
            ret = self.symbol(with_label=True, with_annotation=False, with_order=not _remove_order) \
                  + (self.suffix_sep + "NC" if len(children) == 0 else "") + (self.suffix_sep + "LS" if _is_last else "")
            ret += "" if len(children) == 0 else " " + " ".join(children)
        if mode == "ann":
            _rec_arg = True if _rec_arg is None else _rec_arg
            stacks = [self.symbol(with_annotation=True, with_label=True, with_order=not _remove_order)
                      + ((self.suffix_sep + self.last_suffix) if (_rec_arg is True and not self.name in [self.root_symbol, self.none_symbol]) else "")]
            if len(children) > 0:
                last_child = [False] * (len(children) - 1) + [True]
                children_stacks = [child.pp(mode=mode, arbitrary=arbitrary, _rec_arg=recarg, _top_rec=False, _remove_order=_remove_order)
                                   for child, recarg in zip(children, last_child)]
                for i in range(max([len(child_stack) for child_stack in children_stacks])):
                    acc = []
                    for j in range(len(children_stacks)):
                        if len(children_stacks[j]) > i:
                            acc.append(children_stacks[j][i])
                    acc = " ".join(acc)
                    stacks.append(acc)
            if not _top_rec:
                return stacks
            ret = " ".join(stacks)
        elif mode == "tree":
            direction = "root" if _top_rec else _rec_arg
            if self.num_children > 0:
                def print_children(_children, _direction):
                    _lines = []
                    _dirs = ["up"] + ["middle"] * (len(_children) - 1) if _direction == "up" \
                        else ["middle"] * (len(_children) - 1) + ["down"]
                    for elem, _dir in zip(_children, _dirs):
                        elemlines = elem.pp(mode="tree", arbitrary=arbitrary, _rec_arg=_dir, _top_rec=False, _remove_order=_remove_order)
                        _lines += elemlines
                    return _lines

                parent = self.symbol(with_label=True, with_annotation=False, with_order=not _remove_order)
                if isinstance(parent, str):
                    parent = unidecode(parent)
                up_children, down_children = children[:len(children)//2], children[len(children)//2:]
                up_lines = print_children(up_children, "up")
                down_lines = print_children(down_children, "down")
                uplineprefix = "│" if direction == "middle" or direction == "down" else "" if direction == "root" else " "
                lines = [uplineprefix + " " * len(parent) + up_line for up_line in up_lines]
                parentprefix = "" if direction == "root" else '┌' if direction == "up" else '└' if direction == "down" else '├' if direction == "middle" else " "
                lines.append(parentprefix + parent + '┤')
                downlineprefix = "│" if direction == "middle" or direction == "up" else "" if direction == "root" else " "
                lines += [downlineprefix + " " * len(parent) + down_line for down_line in down_lines]
            else:
                connector = '┌' if direction == "up" else '└' if direction == "down" else '├' if direction == "middle" else ""
                s = self.symbol(with_annotation=False, with_label=True, with_order=not _remove_order)
                if isinstance(s, str):
                    s = unidecode(s)
                lines = [connector + s]
            if not _top_rec:
                return lines
            ret = "\n".join(lines)
        return ret

    @classmethod
    def build_dict_from(cls, trees):
        """ build dictionary for collection of trees (Nodes)"""
        allnames = set()
        alllabels = set()
        suffixes = [cls.suffix_sep + cls.leaf_suffix, cls.suffix_sep + cls.last_suffix,
                    "{}{}{}{}".format(cls.suffix_sep, cls.leaf_suffix, cls.suffix_sep, cls.last_suffix)]
        for tree in trees:
            treenames, treelabels = tree._all_names_and_labels()
            allnames.update(treenames)
            alllabels.update(treelabels)
        if len(alllabels) == 1 and alllabels == {None} or len(alllabels) == 0:
            alltokens = allnames
        else:
            alltokens = set(sum([[token+"/"+label for label in alllabels] for token in allnames], []))

        indic = OrderedDict([("<MASK>", 0), ("<START>", 1), ("<STOP>", 2),
                             (cls.root_symbol, 3), (cls.none_symbol, 4)])
        outdic = OrderedDict()
        outdic.update(indic)
        offset = len(indic)
        alltokens = ["<RARE>"] + sorted(list(alltokens))
        numtokens = len(alltokens)
        newidx = 0
        for token in alltokens:
            indic[token] = newidx + offset
            newidx += 1
        numtokens = len(alltokens)
        newidx = 0
        for token in alltokens:
            outdic[token] = newidx + offset
            for i, suffix in enumerate(suffixes):
                outdic[token +
                       suffix] = newidx + offset + (i + 1) * numtokens
            newidx += 1
        return indic, outdic

    def _all_names_and_labels(self):
        names = set()
        labels = set()
        names.add(self.name)
        for child in self.children:
            childnames, childlabels = child._all_names_and_labels()
            names.update(childnames)
            labels.update(childlabels)
        return names, labels

    def equals(self, other):
        if other is None:
            return False
        ret = self._eq_rec(other)
        return ret

    def __eq__(self, other):
        return self.equals(other)

    def _eq_rec(self, other, _self_pos_in_list=None, _other_pos_in_list=None):
        if isinstance(other, list):
            print("other is list")
            return False
        same = self.name == other.name
        same &= self.label == other.label
        # same &= self.order == other.order
        same &= _self_pos_in_list == _other_pos_in_list
        ownchildren = self.children + tuple()
        otherchildren = other.children + tuple()
        j = 0
        while j < len(ownchildren):
            child = ownchildren[j]
            order_matters = False
            if child.order is not None:
                order_matters = True
            found = False
            i = 0
            while i < len(otherchildren):
                otherchild = otherchildren[i]
                if otherchild.order is not None:
                    order_matters = True
                equality = child._eq_rec(otherchild, _self_pos_in_list=j, _other_pos_in_list=i) \
                    if order_matters else child._eq_rec(otherchild)
                if equality:
                    found = True
                    break
                i += 1
            if found:
                otherchildren = otherchildren[:i] + otherchildren[i+1:]
                ownchildren = ownchildren[:j] + ownchildren[j+1:]
            else:
                j += 1
            same &= found
        same &= len(otherchildren) == 0 and len(ownchildren) == 0   # all children must be matched
        return same