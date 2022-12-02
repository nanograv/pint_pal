import itertools
import logging
import os.path
import subprocess
import textwrap
import tempfile
from collections import defaultdict
from io import StringIO

import matplotlib.pyplot as plt
from IPython.display import Markdown, display


class Report:
    """A report that is gradually built up.
    
    The idea is that an object like this is created at the beginning of
    a notebook, and various cells report various things, generating plots
    and writing markdown/math output both to the report and the notebook
    cells. At the end the report can be rendered to PDF.
    
    The report is organized into "sections"; each section is appended to
    by successive cells, and the sections are emitted in the order they 
    were first encountered. The intention is that sections can have a
    short name to refer to them by and a long name that appears in the
    document.
    
    This could probably be expanded to include page headers and footers
    for the final PDF; PDF generation goes through pandoc, which is quite
    configurable (and HTML might allow for such things too).
    """

    def __init__(self, *, title="", sections=None, figure_location=None):
        self.title = title
        self.sections = [] if sections is None else sections
        self.section_titles = {}
        self.section_content = defaultdict(StringIO)
        if figure_location is None:
            self.temporary_directory_object = tempfile.TemporaryDirectory()
            # Will go away when (if) this report is garbage collected
            figure_location = self.temporary_directory_object.name
        self.figure_location = figure_location

    def _ensure_section(self, section):
        if section not in self.sections:
            self.sections.append(section)

    def new_section(self, section):
        self._ensure_section(section)
        self.section_content[section] = StringIO()

    def add_markdown(self, section, content, *, also_display=True):
        self._ensure_section(section)
        if also_display:
            display(Markdown(content))
        self.section_content[section].write("\n\n")
        self.section_content[section].write(content)

    def add_verbatim(
        self, section, content, *, highlight_language="", also_display=True
    ):
        new_content = f"```{highlight_language}\n{content}\n```\n"
        self.add_markdown(section, new_content, also_display=also_display)

    # I'm not sure about tables - Markdown tables are not too annoying
    # and it's not super clear what format users would pass us tables in
    # (Astropy Tables? lists of lists they'd have to make themselves?)
    # https://www.markdownguide.org/extended-syntax/#tables

    def _new_figure_filename(self, section):
        for i in itertools.count(start=1):
            name = os.path.join(self.figure_location, f"figure-{section}-{i:04d}.png")
            if not os.path.exists(name):
                return name

    def add_plot(
        self,
        section,
        figure=None,
        *,
        caption=None,
        also_display=False,
        **savefig_kwargs,
    ):
        self._ensure_section(section)
        # Include caption?
        filename = self._new_figure_filename(section)
        if figure is None:
            figure = plt.gcf()
        figure.savefig(filename, dpi=300, **savefig_kwargs)
        if caption is None:
            caption = filename
        new_content = f"![{caption}]({filename})\n\n"
        self.add_markdown(section, new_content, also_display=also_display)

    def generate(self, *, include_title=True):
        with StringIO() as o:
            if self.title and include_title:
                print(f"# {self.title}\n\n", file=o)
            for s in self.sections:
                t = self.section_titles.get(s, s)
                print(f"\n\n## {t}\n\n", file=o)
                print(self.section_content[s].getvalue(), file=o)
            return o.getvalue()

    def generate_pdf(self, pdf_filename):
        subprocess.run(
            [
                "pandoc",
                "--from",
                "markdown",
                "--to",
                "pdf",
                "--metadata",
                f"title={self.title}",
                "-o",
                pdf_filename,
                "--pdf-engine",
                "weasyprint",
            ],
            text=True,
            input=self.header + self.generate(include_title=False),
        )

    # We will display bold in red
    # The page margins are supplemented by the margins in the web HTML
    header = textwrap.dedent(
        """
        ---
        header-includes: |
          <style>
          body {
            max-width: 70em;
            background-color: #f0f0ff;
          }
          @media print {
            body {
                background-color: transparent;
                font-size: 10pt;
            }
          }
          @page {
            size: letter;
            margin: 1cm;
          }
          strong {
            color: #ff0000;
          }
          </style>
        ---
        """
    )

    def generate_html(self, html_filename):
        # The HTML is constructed from a template that can be viewed with `pandoc -D html`
        subprocess.run(
            [
                "pandoc",
                "--from",
                "markdown",
                "--to",
                "html",
                "-s",  # standalone, i.e., complete HTML including <html>
                "--self-contained",  # include images inline
                "--mathjax",
                "--metadata",
                f"title={self.title}",
                "-o",
                html_filename,
            ],
            text=True,
            input=self.header + self.generate(include_title=False),
        )

    def begin_capturing_log(self, section, introduction="", *, level=logging.WARNING):
        self._ensure_section(section)
        print(introduction +"\n", file=self.section_content[section])
        # This call ensures that logging is fully initialized before we add things
        # Without this the notebook loses the log messages
        logging.debug("Starting log capturing to report")
        report_log = logging.StreamHandler(self.section_content[section])
        report_log.setLevel(level)
        report_log.setFormatter(
            logging.Formatter("- `%(name)s`: %(levelname)s - %(message)s")
        )
        logging.getLogger().addHandler(report_log)
