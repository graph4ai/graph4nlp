import xml.etree.ElementTree as Et
from collections import defaultdict


class Triple:

    def __init__(self, s, p, o):
        self.s = s
        self.o = o
        self.p = p


class Tripleset:

    def __init__(self):
        self.triples = []

    def fill_tripleset(self, t):
        for xml_triple in t:
            s, p, o = xml_triple.text.split(' | ')
            triple = Triple(s, p, o)
            self.triples.append(triple)


class Lexicalisation:

    def __init__(self, lex, comment, lid):
        self.lex = lex
        self.comment = comment
        self.id = lid


class Entry:

    def __init__(self, category, size, eid):
        self.originaltripleset = []
        self.modifiedtripleset = Tripleset()
        self.lexs = []
        self.category = category
        self.size = size
        self.id = eid

    def fill_originaltriple(self, xml_t):
        otripleset = Tripleset()
        self.originaltripleset.append(otripleset)   # multiple originaltriplesets for one entry
        otripleset.fill_tripleset(xml_t)

    def fill_modifiedtriple(self, xml_t):
        self.modifiedtripleset.fill_tripleset(xml_t)

    def create_lex(self, xml_lex):
        comment = xml_lex.attrib['comment']
        lid = xml_lex.attrib['lid']
        lex = Lexicalisation(xml_lex.text, comment, lid)
        self.lexs.append(lex)

    def count_lexs(self):
        return len(self.lexs)


class Benchmark:

    def __init__(self):
        self.entries = []

    def fill_benchmark(self, fileslist):
        # print(fileslist)
        for file in fileslist:
            print(file)
            tree = Et.parse(file[0] + '/' + file[1])
            root = tree.getroot()
            for xml_entry in root.iter('entry'):
                # ignore triples with no lexicalisations
                lexfound = False
                for child in xml_entry:
                    if child.tag == "lex":
                        lexfound = True
                        break
                if lexfound is False:
                    continue

                entry_id = xml_entry.attrib['eid']
                category = xml_entry.attrib['category']
                size = xml_entry.attrib['size']
                entry = Entry(category, size, entry_id)
                for child in xml_entry:
                    if child.tag == 'originaltripleset':
                        entry.fill_originaltriple(child)
                    elif child.tag == 'modifiedtripleset':
                        entry.fill_modifiedtriple(child)
                    elif child.tag == 'lex':
                        entry.create_lex(child)
                self.entries.append(entry)

    def total_lexcount(self):
        count = [entry.count_lexs() for entry in self.entries]
        return sum(count)

    def unique_p(self):
        properties = [triple.p for entry in self.entries for triple in entry.modifiedtripleset.triples]
        return len(set(properties))

    def entry_count(self, size=None, cat=None):
        """
        calculate the number of entries in benchmark
        :param size: size (should be string)
        :param cat: category
        :return: entry count
        """
        if not size and cat:
            entries = [entry for entry in self.entries if entry.category == cat]
        elif not cat and size:
            entries = [entry for entry in self.entries if entry.size == size]
        elif not size and not cat:
            return len(self.entries)
        else:
            entries = [entry for entry in self.entries if entry.category == cat and entry.size == size]
        return len(entries)

    def lexcount_size_category(self, size='', cat=''):
        count = [entry.count_lexs() for entry in self.entries if entry.category == cat and entry.size == size]
        return len(count)

    def property_map(self):
        mprop_oprop = defaultdict(set)
        for entry in self.entries:
            for tripleset in entry.originaltripleset:
                for i, triple in enumerate(tripleset.triples):
                    mprop_oprop[entry.modifiedtripleset.triples[i].p].add(triple.p)
        return mprop_oprop
