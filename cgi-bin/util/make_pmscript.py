import sys

vecdir = "/srv/vecs/"

def escape(s, chars):
    for char in chars:
        s = s.replace(char, '\\' + char)
    return s

def wrap_lemma(l):
    def expand_star(single_lemma):
        def wrap_nonempty_with_curlies(part):
            if part == '':
                return part
            return '{' + escape(part, '}') + '}'
        replaced = single_lemma.replace('*', '[\\"\\t"]*')
        otherparts = replaced.split('[\\"\\t"]*')
        with_stringparts = '[\\"\\t"]*'.join(map(wrap_nonempty_with_curlies, otherparts))
        return '[\\["\\t"|"\\n"]]* "\\t"' + with_stringparts + ' "\\t" [\\["\\t"|"\\n"]]*'
    def wrap_pmatchcode(s):
        return '[\\["\\t"|"\\n"]]* "\\t" ' + s + ' "\\t" [\\["\\t"|"\\n"]]*'
    def wrap(s):
        if '*' in s:
            return expand_star(s)
        if (s.startswith('Like(') or s.startswith('Unlike(')) and s.endswith(')'):
            return wrap_pmatchcode(s[0] + s[1:].lower())
        return '[\\["\\t"|"\\n"]]* "\\t" {' + escape(s, '}') + '} "\\t" [\\["\\t"|"\\n"]]*'
    def split_and_wrap(s):
        retval = []
        i = 0
        while True:
            if '[' not in s[i:] or ']' not in s[i:]:
                retval += map(wrap, filter(lambda x: x.strip() != '', s[i:].split(" ")))
                break
            if ' ' not in s[i:]:
                stripped = s[i:].strip()
                if stripped.startswith('[') and stripped.endswith(']'):
                    retval.append(wrap_pmatchcode(stripped))
                elif stripped != '':
                    retval.append(wrap(stripped))
                break
            space_idx = s.find(" ", i)
            rightbrack_idx = s.find("]", i)
            if not s.startswith('[', i):
                stripped = s[i:space_idx].strip()
                if stripped != '':
                    retval.append(wrap(s[i:space_idx]))
                i = space_idx + 1
                continue
            else:
                retval.append(wrap_pmatchcode(s[i:rightbrack_idx + 1]))
                i = rightbrack_idx + 1
                continue
        return retval
    return ' "\\n"'.join(split_and_wrap(l))

def indent(n):
    return lambda x: n * " " + x

def make_script(lemmalist, specs = None):
    vectorfile = vecdir + "all_vec.bin"
    if specs != None:
        if "vectorfile" in specs:
            vectorfile = vecdir + specs["vectorfile"]
    scriptstring = 'set need-separators off\n'
    if 'Like(' in lemmalist or 'Unlike(' in lemmalist:
        scriptstring += '@vec"' + vectorfile + '"\n\n'
    scriptstring += 'regex\n'
    expecting_tag = True
    thistag = "LemmaMatch"
    taglemmas = {}

    for line in lemmalist.split('\n'):
        line = line.strip()
        if line == "":
            expecting_tag = True
            continue
        if expecting_tag:
            thistag = line.replace('@', 'PMATCH_AT')
            expecting_tag = False
            continue
        taglemmas.setdefault(thistag, []).append(line)

    sections = []
    section = ""
    for tag in taglemmas:
        section += (' [ [\n')
        section += ('    [\n')
        section += (' |\n'.join(map(indent(6), map(wrap_lemma, taglemmas[tag]))))
        section += ('\n    ]\n')
        section += ('    EndTag(' + tag + ')\n')
        section += ('  ] ]\n')
        sections.append(section)
        section = ""
    scriptstring += ' |\n'.join(sections)
    scriptstring += ";"
    return scriptstring

if __name__ == "__main__":
    f = open(sys.argv[1], "r")
    print(make_script(f.read()))

