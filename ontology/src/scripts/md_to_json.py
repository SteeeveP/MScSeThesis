"""Module to convert Whimsical Mindmap Markdown to APE ontology."""


import json
import re
from typing import Dict, Iterable, List, Set, Tuple


TYPE_DIM_SHORT_DICT = {
    'DataState': 'State',
    'StatisticalRelevance': 'Relevance',
    'DataSetIndex': 'Index',
}
"""Mapping from type dimension to short name."""


# =============================================================================
#                               Markdown to Graph
# =============================================================================

def read_md_to_dict(file: str) -> Dict[str, Dict[str, Dict]]:
    """Whimsical Mindmap Markdown copy to Dictionary.

    Args:
        file (str): path to md file

    Returns:
        Dict[str, Dict[str, Dict]]: Raw dictionary output
    """
    with open(file, 'r', encoding='utf-8') as tool_file:
        entries = tool_file.read().split('\n')[1:]

    tax_dict: Dict[str, Dict[str, Dict]] = {}
    stack: List[Tuple[int, Dict]] = [(-1, tax_dict)]

    for line in entries:
        line = line.rsplit('- ', maxsplit=1)
        if len(line[0]) > stack[-1][0]:
            stack[-1][1][line[1]] = {}
            stack = stack + [(len(line[0]), stack[-1][1][line[1]])]
        elif len(line[0]) == stack[-1][0]:
            stack[-2][1][line[1]] = {}
            stack = stack[:-1] + [(len(line[0]), stack[-2][1][line[1]])]
        else:
            for i in range(1, len(stack)+1):
                if stack[-i][0] < len(line[0]):
                    stack[-i][1][line[1]] = {}
                    stack = stack[:1-i] + \
                        [(len(line[0]), stack[-i][1][line[1]])]
                    break
    return tax_dict


def resolve_inheritance(taxonomy: Dict[str, Dict]) -> Dict[str, Dict]:
    """Resolves multiple inheritance in type taxonomy.

    Args:
        type_tax (Dict[str, Dict]): Taxonomy with multiple inheritance.

    Returns:
        Dict[str, Dict]: Taxonomy with multiple inheritance resolved.

    Example:
    >>> taxonomy = {
            'A': {
                'C': {},
                'B': {
                    'E': {},
                },
            },
            'B': {
                'D': {},
            },
        }
    >>> resolve_inheritance(taxonomy)
    {
        'A': {
            'C': {},
            'E': {},
        },
        'B': {
            'D': {},
            'E': {},
        },
    }
    """
    # Each item is tuple of parent keys (multiple inheritance) and current node
    queue: List[Tuple[List[str], Dict[str, Dict]]] = [(['root'], taxonomy)]
    # All found nodes with recursive references to children
    # ({'parent': {'child': X}, 'child': X})
    found: Dict[str, Dict] = {'root': {}}

    # BFS: building up inheritance tree in found
    while queue:
        parents, node = queue.pop(0)
        # add each child in node to found
        for child_key, child in node.items():
            # if not yet found
            if child_key not in found:
                # register child
                found[child_key] = {}
                # add child to parents
                for parent_key in parents:
                    found[parent_key][child_key] = found[child_key]
                # add child to queue if not terminal
                if child:
                    queue.append(([child_key], child))
            # already found and not terminal, multiple inheritance
            elif child:
                # add item back to queue with new parent
                queue.append((parents + [child_key], child))
            # already found and terminal
            else:
                # add child to parents
                for parent_key in parents:
                    found[parent_key][child_key] = found[child_key]
    # DFS: return deep copy of found
    deep_copy = {}
    stack = [(deep_copy, found['root'])]
    while stack:
        copy_ref, node = stack.pop()
        for child_key, child in node.items():
            # copy step: add str key to dict
            copy_ref[child_key] = {}
            if child:
                stack.append((copy_ref[child_key], child))
    return deep_copy


# =============================================================================
#                               Types
# =============================================================================

def tax_dict_to_tuples(
    tax_dict: Dict[str, Dict],
    root: str,
) -> Tuple[
        Set[Tuple[str, str]],
        Set[str], Set[str],
]:
    """Converts taxonomy dict to set of tuples.

    Also returns set of terminal and non-terminal nodes.
    > Requires resolved inheritance. (Non recursive)

    Args:
        tax_dict (Dict[str, Dict]): Taxonomy dict.
        root (str): Root node of taxonomy.

    Returns:
        Tuple[Set[Tuple[str, str]], Set[str], Set[str]]:
            Set of tuples, set of terminal nodes, set of non-terminal nodes.
    """
    tuples = set()
    found = set()
    non_terminal = set()
    stack = [(root, tax_dict)]
    while stack:
        parent, children = stack.pop()
        found.add(parent)
        if children:
            non_terminal.add(parent)
            for child in children:
                tuples.add((parent, child))
        stack.extend(children.items())
    return tuples, found-non_terminal, non_terminal


def get_object_flag_dict_from_md(file) -> Dict[str, str]:
    """Reads object flag types from md file
    and returns a dictionary of object to object flag.
    """
    with open(file, 'r', encoding='utf-8') as tool_file:
        entries = tool_file.read().split('\n')[1:]
    object_flag_types = {}
    curr = None
    for entry in entries:
        if entry.startswith('-'):
            curr = entry[2:]
        else:
            assert curr is not None
            object_flag_types[entry.strip()[2:]] = curr
    return object_flag_types


def get_constants(
    type_dims: List[str],
    type_dim_file_dict: Dict[str, str],
) -> Tuple[
    Dict[str, Dict[str, Dict]],
    Dict[str, Set[str]],
    Dict[str, Set[str]],
    Dict[str, str],
    Dict[str, str],
    Dict[str, Set[Tuple[str, str]]],
]:
    """Returns constants for type taxonomy."""
    flag_suffix = {
        dim: TYPE_DIM_SHORT_DICT[dim]
        for dim
        in type_dims
        if dim != 'DataClass'
    }
    type_taxes = {
        dim: read_md_to_dict(type_dim_file_dict[dim])
        for dim
        in type_dims
    }
    for dim, tax in type_taxes.items():
        # resolve multiple inheritance
        type_taxes[dim] = resolve_inheritance(tax)
        # add default type to flag types
        if dim != 'DataClass':
            for key in type_taxes[dim]:
                if not key.startswith('No') and key.endswith(flag_suffix[dim]):
                    type_taxes[dim][key].update({'No'+flag_suffix[dim]: {}})
    # Add graph info to taxonomies
    rel_tuples: Dict[str, Set[Tuple[str, str]]] = {}
    term_types: Dict[str, Set[str]] = {}
    concept_types: Dict[str, Set[str]] = {}
    for dim in type_dims:
        rel_tuples[dim], term_types[dim], concept_types[dim] = tax_dict_to_tuples(
            type_taxes[dim],
            dim,
        )
    # Add object flag types
    object_flag_types = get_object_flag_dict_from_md(
        type_dim_file_dict['DataClass'])
    return type_taxes, term_types, concept_types, object_flag_types, flag_suffix, rel_tuples


# =============================================================================
#                               Tools
# =============================================================================

def get_sub_terms(dim: str, concept: str, type_taxes: Dict) -> List[str]:
    """Recursion wrapper to get terminal nodes of conceptual taxonomy subtree."""
    def get_terms(tax: Dict[str, Dict[str, Dict]]) -> List[str]:
        terms = []
        for key, val in tax.items():
            if not len(val):
                terms.append(key)
            else:
                terms += get_terms(val)
        return terms

    def get_sub_tax_terms(tax: Dict[str, Dict[str, Dict]], concept: str) -> List[str]:
        try:
            return get_terms(tax[concept])
        except KeyError as err:
            for val in tax.values():
                try:
                    return get_sub_tax_terms(val, concept)
                except KeyError:
                    pass
            raise KeyError from err
    return get_sub_tax_terms(type_taxes[dim], concept)


def get_sub_nodes(dim: str, concept: str, type_taxes: Dict) -> List[str]:
    """Recursion wrapper to get all sub nodes of conceptual taxonomy subtree."""
    def get_nodes(tax: Dict[str, Dict[str, Dict]]) -> List[str]:
        terms = []
        for key, val in tax.items():
            terms.append(key)
            if len(val):
                terms += get_nodes(val)
        return terms

    def get_sub_tax_nodes(tax: Dict[str, Dict[str, Dict]], concept: str, found: List[str]) -> List[str]:
        found += list(tax.keys())
        try:
            #! temp: since multiple inheritance is done with otherwise empty sub taxes for now
            try:
                assert len(tax[concept]) > 0
            except AssertionError as err:
                raise KeyError from err
            return [t for t in get_nodes(tax[concept]) if not t in found]
        except KeyError as err:
            for val in tax.values():
                try:
                    return get_sub_tax_nodes(val, concept, found)
                except KeyError:
                    pass
            raise KeyError from err

    return get_sub_tax_nodes(type_taxes[dim], concept, [])


def get_term_combinations(
    inp: List[Dict[str, str]],
    out: List[Dict[str, str]],
    fill_dict: Dict[Tuple[str, str], Iterable[str]],
    remaining_keys: List[Tuple[str, str]],
    curr_comb: Dict[Tuple[str, str], str],
) -> List[Tuple[List[Dict[str, str]], List[Dict[str, str]]]]:
    """Another recursive function to return all combinations of fill values.

    Args:
        inp (List[Dict[str, str]]): input params
        out (List[Dict[str, str]]): output params
        fill_dict (Dict[Tuple[str, str], Iterable[str]]): fill value dictionary
        remaining_keys (List[Tuple[str, str]]): not yet assigned keys
        curr_comb (Dict[Tuple[str, str], str]): current assignment

    Returns:
        List[Tuple[List[Dict[str, str]], List[Dict[str, str]]]]: List of all assignments

    """
    if remaining_keys:
        modes = []
        for val in fill_dict[remaining_keys[0]]:
            modes += get_term_combinations(
                inp, out,
                fill_dict, remaining_keys[1:],
                curr_comb.copy() | {remaining_keys[0]: val},
            )
        return modes
    inp_ = []
    for param in inp:
        new_param = {}
        for dim, type in param.items():
            try:
                new_param[dim] = curr_comb[(dim, type)]
            except KeyError:
                new_param[dim] = type
        inp_.append(new_param)
    out_ = []
    for param in out:
        new_param = {}
        for dim, type in param.items():
            try:
                new_param[dim] = curr_comb[(dim, type)]
            except KeyError:
                new_param[dim] = type
        out_.append(new_param)
    return [(inp_, out_)]


def filter_flag_combinations(
    combs: List[Tuple[List[Dict[str, str]], List[Dict[str, str]]]],
    type_dims: List[str],
    object_flag_types: Dict[str, str],
    flag_suffix: Dict[str, str],
    type_taxes: Dict[str, Dict[str, Dict]],
) -> List[Tuple[List[Dict[str, str]], List[Dict[str, str]]]]:
    """Removes invalid flag combinations.

    Args:
        combs (List[Tuple[List[Dict[str, str]], List[Dict[str, str]]]]): List of combinations

    Returns:
        List[Tuple[List[Dict[str, str]], List[Dict[str, str]]]]: List of valid combinations

    """
    res_ix: List[int] = []
    for i, (inp, out) in enumerate(combs):
        try:
            for param in inp + out:
                for dim in type_dims:
                    if dim != 'DataClass':
                        try:
                            assert (
                                param[dim] == object_flag_types[param['DataClass']
                                                                ]+flag_suffix[dim]
                                or param[dim] == dim
                                or param[dim] == 'No'+flag_suffix[dim]
                                or param[dim] in get_sub_nodes(
                                    dim,
                                    object_flag_types[param['DataClass']
                                                      ]+flag_suffix[dim],
                                    type_taxes,
                                )
                            )
                        except KeyError:
                            pass
            res_ix.append(i)
        except AssertionError:
            pass
    return [combs[ix] for ix in res_ix]


def map_generics_to_term(
    inp: List[Dict[str, str]],
    out: List[Dict[str, str]],
    object_flag_types: Dict[str, str],
    flag_suffix: Dict[str, str],
    term_types: Dict[str, Set[str]],
    type_taxes: Dict[str, Dict[str, Dict]],
    type_dims: List[str],
) -> List[Tuple[
    List[Dict[str, str]],
    List[Dict[str, str]],
]]:
    """Maps input and output with generic types to terminal type params.

    Args:
        inp (List[Dict[str, str]]): List of input params
        out (List[Dict[str, str]]): List of output params

    Returns:
        List[Tuple[ List[Dict[str, str]], List[Dict[str, str]], ]]: List of input output tuples w/o generics.

    """
    #! temp remove "\\" suffix
    for param in inp + out:
        for key, value in param.items():
            param[key] = re.sub(r'\\.*', '', value)

    # fill empty dims with default:
    for param in inp:
        stripped_cls = re.sub(r'^.*:', '', param['DataClass'])
        for key, value in param.items():
            if value == "":
                # param[key] = key
                assert key != 'DataClass'
                param[key] = object_flag_types[stripped_cls]+flag_suffix[key]
    for param in out:
        for key, value in param.items():
            if value == "":
                param[key] = f'No{flag_suffix[key]}'

    # get (dim, generic_key, term_list) tuples
    fill_values: Dict[Tuple[str, str], Iterable[str]] = {}
    for dim, dim_terms in term_types.items():
        for param in inp:
            if param[dim] in ['A', 'B']:
                if dim != 'DataClass':
                    try:
                        fill_values.update({
                            (dim, param[dim]): get_sub_terms(
                                dim,
                                object_flag_types[param['DataClass']
                                                  ]+flag_suffix[dim],
                                type_taxes
                            )
                        })
                    except KeyError:
                        fill_values.update(
                            {(dim, param[dim]): ['No'+flag_suffix[dim]]})
                else:
                    fill_values.update({(dim, param[dim]): dim_terms})
            elif param[dim].startswith('A:'):
                [super_type] = param[dim].split(':')[1:]
                if any(':' in x for x in get_sub_terms(dim, super_type, type_taxes)):
                    raise ValueError(
                        str(super_type) + '\t' + str(get_sub_terms(dim, super_type, type_taxes)))
                fill_values.update({
                    (dim, param[dim]): get_sub_terms(dim, super_type, type_taxes)
                })
        # if len(inp) == 0:
        for param in out:
            if (dim, param[dim]) not in fill_values:
                if param[dim] in ['A', 'B']:
                    if dim != 'DataClass':
                        try:
                            fill_values.update({
                                (dim, param[dim]): get_sub_terms(
                                    dim,
                                    object_flag_types[param['DataClass']
                                                      ]+flag_suffix[dim],
                                    type_taxes
                                )
                            })
                        except KeyError:
                            fill_values.update(
                                {(dim, param[dim]): ['No'+flag_suffix[dim]]})
                    else:
                        fill_values.update({(dim, param[dim]): dim_terms})
                elif param[dim].startswith('A:'):
                    fill_values.update({
                        (dim, param[dim]): get_sub_terms(dim, param[dim].split(':')[1], type_taxes)
                    })

    term_combinations = get_term_combinations(
        inp,
        out,
        fill_values,
        list(fill_values.keys()),
        {},
    )
    filtered_term_combinations = filter_flag_combinations(
        term_combinations,
        type_dims,
        object_flag_types,
        flag_suffix,
        type_taxes,
    )
    if len(filtered_term_combinations) == 0:
        print(term_combinations)
        raise ValueError(
            f'No valid term combinations found for {inp} and {out}')
    return filtered_term_combinations


def traverse_tool_tax(
    tax: Dict[str, Dict[str, Dict]],
    type_dims: List[str],
    object_flag_types: Dict[str, str],
    flag_suffix: Dict[str, str],
    term_types: Dict[str, Set[str]],
    type_taxes: Dict[str, Dict[str, Dict]],
) -> Tuple[
    Dict[str, Dict[str, Dict]],
    Dict[str, Dict[str, Dict[str, List[Dict[str, str]]]]],
]:
    """Split tools and their annotations.

    Args:
        tax (Dict[str, Dict[str, Dict]]): Tool taxonomy dict

    Returns:
        Dict[str, Dict[str, Dict]]: Pruned tool taxonomy
        Dict[str, Dict[str, Dict[str, List[Dict[str, str]]]]]: List of tool modes and their annotations

    """
    new_dict: Dict[str, Dict[str, Dict]] = {}
    tool_dict: Dict[str, Dict[str, Dict[str, List[Dict[str, str]]]]] = {}

    default_params = ['DataClass', 'DataState',
                      'StatisticalRelevance', 'DataSetIndex']

    for key, value in tax.items():
        key = key.strip()
        if any("->" in key_ and key_.strip() != '->' for key_ in value.keys()):
            # if not key.startswith('pair') and key.endswith('plot') and any('(Column,' in key_ for key_ in value.keys()):
            #     tmp_vals = sum([
            #         [
            #             (
            #                 key_.replace('(Column,', f'({repl},'),
            #                 val_,
            #             )
            #             for repl
            #             in [
            #                 'StrColumn',
            #                 'DateTimeColumn',
            #                 'BoolColumn',
            #                 'MixedColumn',
            #             ]
            #         ] if '(Column,' in key_ else [(key_, val_)]
            #         for key_, val_
            #         in value.items()
            #     ], [])
            #     value = dict(tmp_vals)
            modes: List[Dict[str, List[Dict[str, str]]]] = []
            for mode in value.keys():
                try:
                    inp, out = re.sub(r'[\s{}(]', '', mode).replace('),', ')').split(
                        "->")
                except ValueError as v_err:
                    raise ValueError(mode) from v_err
                inp_ = [
                    {k: v for k, v in zip(
                        default_params, param.split(',')) if k in type_dims}
                    for param
                    in re.sub(r'\)$', '', inp).split(')')
                    if param != ""
                ] if len(inp) > 0 else []
                out_ = [
                    {k: v for k, v in zip(
                        default_params, param.split(',')) if k in type_dims}
                    for param
                    in re.sub(r'\)$', '', out).split(')')
                ] if len(out) > 0 else []
                try:
                    modes += [
                        {"input": inp_term, "output": out_term}
                        for inp_term, out_term
                        in map_generics_to_term(
                            inp_,
                            out_,
                            object_flag_types,
                            flag_suffix,
                            term_types,
                            type_taxes,
                            type_dims,
                        )
                    ]
                except KeyError as err:
                    raise KeyError(key + ' : ' + inp + ' -> ' + out) from err
            if len(modes) > 1:
                # new_dict[key] = {f'{key}_{i}': {} for i in range(len(modes))}
                new_dict[key] = {key: {}}
                tool_dict[key] = {f'{key}_{i}': mode for i,
                                  mode in enumerate(modes)}
            else:
                new_dict[key] = {key: {}}
                try:
                    tool_dict[key] = {f'{key}_0': modes[0]}
                except IndexError as err:
                    raise IndexError(str(key) + str(value)) from err
        elif any(key_.strip() != '->' for key_ in value.keys()):
            new_dict[key], tool_dict_ = traverse_tool_tax(
                value,
                type_dims,
                object_flag_types,
                flag_suffix,
                term_types,
                type_taxes,
            )
            tool_dict.update(tool_dict_)
    return new_dict, tool_dict


#! hotfix
def remove_modes_tool_tax(tool_tax: Dict[str, Dict]) -> Dict[str, Dict]:
    """Remove modes from tool taxonomy."""
    new_tool_tax = {}
    for tool, sub_tools in tool_tax.items():
        if '->' not in tool:
            new_tool_tax[tool] = remove_modes_tool_tax(sub_tools)
    return new_tool_tax


def tool_annotations_to_json(
        tool_annotations_dict: Dict[str, Dict[str, Dict[str, List[Dict[str, str]]]]],
        target_file: str,
) -> None:
    """Wrapper to write tool annotation dictionary into json file.

    Args:
        tool_annotations_dict (Dict[str, Dict[str, Dict[str, List[Dict[str, str]]]]]): Tool annotation dictionary
        target_file (str): Path to output file

    """
    functions = []
    for tool, modes in tool_annotations_dict.items():
        for mode, params in modes.items():
            functions.append({
                'id': mode,
                'label': mode,
                'taxonomyOperations': [tool],
                'inputs': [{dim: [type] for dim, type in inp.items()} for inp in params['input']],
                'outputs': [{dim: [type] for dim, type in inp.items()} for inp in params['output']],
            })
    with open(target_file, 'w', encoding='utf-8') as json_file:
        json.dump({'functions': functions}, json_file, indent=4)


def tuples_to_owl(
    type_tuples: List[Tuple[str, str]],
    tool_tuples: List[Tuple[str, str]],
    target_file: str,
    prefix: str = 'http://www.co-ode.org/ontologies/ont.owl#',
) -> None:
    with open(target_file, 'w', encoding='utf-8') as owl_file:
        # header
        owl_file.write("""<?xml version="1.0"?>
<rdf:RDF xmlns="http://www.co-ode.org/ontologies/ont.owl#"
     xml:base="http://www.co-ode.org/ontologies/ont.owl"
     xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
     xmlns:owl="http://www.w3.org/2002/07/owl#"
     xmlns:xml="http://www.w3.org/XML/1998/namespace"
     xmlns:xsd="http://www.w3.org/2001/XMLSchema#"
     xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#">
    <owl:Ontology rdf:about="http://www.co-ode.org/ontologies/ont.owl"/>



    <!--
    ///////////////////////////////////////////////////////////////////////////////////////
    //
    // Classes
    //
    ///////////////////////////////////////////////////////////////////////////////////////
     -->



""")
        owl_file.write(f'\t<owl:Class rdf:about="{prefix}TypesTaxonomy"/>\n\n')
        for parent, child in type_tuples:
            owl_file.write(f'\t<owl:Class rdf:about="{prefix}{child}">\n')
            owl_file.write(
                f'\t\t<rdfs:subClassOf rdf:resource="{prefix}{parent}"/>\n')
            owl_file.write('\t</owl:Class>\n\n')
        owl_file.write(f'\t<owl:Class rdf:about="{prefix}ToolsTaxonomy"/>\n\n')
        for parent, child in tool_tuples:
            # if child != re.sub(r'_[0-9]+$', '', child):
            owl_file.write(f'\t<owl:Class rdf:about="{prefix}{child}">\n')
            owl_file.write(
                f'\t\t<rdfs:subClassOf rdf:resource="{prefix}{parent}"/>\n')
            owl_file.write('\t</owl:Class>\n\n')
        owl_file.write('</rdf:RDF>\n')
