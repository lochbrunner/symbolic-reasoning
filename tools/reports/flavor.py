import enum


class Flavors(enum.Enum):
    latex = (
        r"\begin{{{}}}",
        r"\end{{{}}}",
        "document",
        """\
\\usepackage[utf8]{{inputenc}}
\\usepackage{{pgfplots}}
\\DeclareUnicodeCharacter{{2212}}{{−}}
\\usepgfplotslibrary{{{pgfplotslibs}}}
\\usetikzlibrary{{{tikzlibs}}}
\\pgfplotsset{{compat=newest}}
""",
    )
    context = (
        r"\start{}",
        r"\stop{}",
        "text",
        """\
\\setupcolors[state=start]
\\usemodule[tikz]
\\usemodule[pgfplots]
\\usepgfplotslibrary[{pgfplotslibs}]
\\usetikzlibrary[{tikzlibs}]
\\pgfplotsset{{compat=newest}}
% groupplot doesn’t define ConTeXt stuff
\\unexpanded\\def\\startgroupplot{{\\groupplot}}
\\unexpanded\\def\\stopgroupplot{{\\endgroupplot}}
""",
    )

    def start(self, what):
        return self.value[0].format(what)

    def end(self, what):
        return self.value[1].format(what)

    def preamble(self, data=None):
        if data is None:
            data = {
                "pgfplots libs": ("groupplots", "dateplot"),
                "tikz libs": ("patterns", "shapes.arrows"),
            }
        pgfplotslibs = ",".join(data["pgfplots libs"])
        tikzlibs = ",".join(data["tikz libs"])
        return self.value[3].format(pgfplotslibs=pgfplotslibs, tikzlibs=tikzlibs)

    def standalone(self, code):
        docenv = self.value[2]
        return f"{self.preamble()}{self.start(docenv)}\n{code}\n{self.end(docenv)}"
