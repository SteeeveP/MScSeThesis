"""Some utility functions to load APE ontologies in OWL format."""


from typing import Set, Tuple, cast

from bs4 import BeautifulSoup, PageElement, ResultSet, Tag


def owl_xml_to_cls_elements(path: str) -> ResultSet[Tag]:
    """Reads owl from xml file and returns list of found class Tags.

    Args:
        path (str): Path to xml file.

    Returns:
        ResultSet[Tag]: All found Tags with name 'owl:Class'.
    """ 
    with open(path, 'r', encoding='utf-8') as owl_f:
        xml_str = owl_f.read()
    bs_data = BeautifulSoup(xml_str, 'xml')
    if bs_data.RDF is None:
        raise TypeError('Invalid format, RDF tag not direct child.')
    if path.endswith('.owl'):
        return bs_data.RDF.find_all('owl:Class', recursive=False)
    if path.endswith('rdf'):
        return  bs_data.RDF.find_all('rdf:Description', recursive=False)
    raise TypeError('Invalid file name suffix. Use owl or rdf.')

def cls_elements_to_rel_tuples(tags: ResultSet[Tag], prefix: str) -> Set[Tuple[str, str]]:
    """Iterates through class tags
    and returns tuples of found class to parent class relations.

    Args:
        tags (ResultSet[Tag]): List of class tags

    Raises:
        AttributeError: A found class tag has no `rdf:about` attribute.

    Returns:
        Set[Tuple[str, str]]: Set of found relation tuples.
    """
    rel_tuples: Set[Tuple[str, str]]

    rel_tuples = set()

    def get_cls_from_str(cls_str: str) -> str:
        try:
            return cls_str.split(prefix)[1]
        except IndexError:
            return cls_str.split('#')[-1]

    for item in tags:
        # class name
        try:
            item_cls = get_cls_from_str(item.attrs['rdf:about'])
        except (AttributeError, KeyError) as exc:
            raise AttributeError(
                f'Class item has no rdf:about attribute: {str(item)}.',
            ) from exc
        # CLASSES that are subclasses of data0004 somehow do not add data0004 to the results set?
        # parent class name
        for child in cast(ResultSet[PageElement], item.find_all('rdfs:subClassOf')):
            try:
                item_par_cls = get_cls_from_str(cast(Tag, child).attrs['rdf:resource'])
                rel_tuples.add((item_cls, item_par_cls))
            except KeyError:
                # no direct subclass mention
                try:
                    for nested_cls in cast(Tag, child).find_all('rdf:Description'):
                        item_par_cls = get_cls_from_str(cast(Tag, nested_cls).attrs['rdf:about'])
                        rel_tuples.add((item_cls, item_par_cls))
                except (AttributeError, KeyError):
                    pass
            except AttributeError:
                pass

        # if no tuple added, root class: no rel tuple required
    return rel_tuples


def load_owl(path: str, prefix: str) -> Set[Tuple[str, str]]:
    """Simple wrapper function.

    Args:
        path (str): Path to owl file.

    Raises:
        ValueError: No class-subclass relations found in owl file.

    Returns:
        Set[Tuple[str, str]]: Set of found relation tuples.
    """
    found_tags = owl_xml_to_cls_elements(path)
    if len(found_tags) == 0:
        raise ValueError('No class-subclass relations found in owl file.')
    return cls_elements_to_rel_tuples(found_tags, prefix)
