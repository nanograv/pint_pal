from collections import OrderedDict
import textwrap
import subprocess

class MarkdownSection:
    def __init__(self, title: str = "", content: list | None = None, remove_spacing: bool = False) -> None:
        """
        Initialization for the MarkdownSection

        Parameters
        ----------
        title : str, default=""
           Title of the section.
        content : list | None, default=None
           List of content, either pre-generated as a list or created if None.
        remove_spacing : bool, default=False
            If true, remove excess line spacing in the generated markdown.
        """
        self._title = title
        self._content = [] if content is None else content
        self.remove_spacing = remove_spacing
        
    @property
    def title(self) -> str: 
        return self._title

    @title.setter
    def title(self, value: str) -> None:
        self._title = value
        return

    @property
    def content(self) -> list:
        return self._content

    @content.setter
    def content(self, value: list) -> None:
        self._content = value
        return

    def add_content(self, content: str) -> None:
        self._content.append(content)
        return

    def generate(self, *, include_title=True) -> str:
        """
        Creates the combined markdown string from the content list

        Parameters
        ----------
        include_title : bool, default=True
            If True, includes the title as a section in the output string.
        
        Returns
        -------
        output : str
            Combined markdown string.
        """
        output = ""
        if self._title and include_title:
            output += f"\n\n## {self._title}\n\n"
        for element in self._content:
            if self.remove_spacing:
                output += f"{element}    \n"
            else:
                output += f"{element}\n\n"
        return output

        
class MarkdownReport:
    def __init__(self, title: str = "", sections: OrderedDict | dict | None = None, font_size: int = 10):
        """
        Initialization for the MarkdownReport
        
        Parameters
        ----------
        title : str
            Name of the report.
        sections : OrderedDict | dict | None, default=None
            Sections provided to the MarkdownReport. Does not take MarkdownSection values. If None, an empty OrderedDict is created.
        font_size : int
            Size to render text at.
        """
        self.title = title
        self.sections = OrderedDict() if sections is None else sections
        self.current_section = None if sections is None else sections[-1].title

        self.header = textwrap.dedent(
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
            """
        )
        self.header += f"""      font-size: {font_size}pt;"""
        self.header += textwrap.dedent(
            """
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
        
    def get_section(self, section_title: str) -> MarkdownSection:
        """
        Returns a section of the report.

        Parameters
        ----------
        section_title : str
            Title of the section to return.

        Raises
        ------
        KeyError
            If the section with the given title provided does not exist.
        """
        if section_title not in self.sections.keys():
            raise KeyError(f"Section title not found: {section_title}")
        return self.sections[section_title]

    def add_section(self, section: MarkdownSection, set_current: bool = True) -> None:
        """
        Adds a section to the report given a title string.

        Parameters
        ----------
        section : MarkdownSection
            Pre-generated MarkdownSection to add to the report.
        set_current : bool, default=True
            If true, this section is now the current one at the bottom of the report.
        """
        if set_current:
            self.current_section = section.title()
        return

    def add_section_by_title(self, section_title: str, set_current: bool = True, remove_spacing: bool = False) -> None:
        """
        Adds a section to the report given a title string.

        Parameters
        ----------
        section_title : str
            Title of the new section.
        set_current : bool, default=True
            If true, this section is now the current one at the bottom of the report.
        remove_spacing : bool, default=False,
            If true, remove excess line spacing. Sends to the individual MarkdownSection.
        """
        if section_title in self.sections.keys():
            raise KeyError(f"Section title already exists: {section_title}")

        self.sections[section_title] = MarkdownSection(title=section_title, remove_spacing=remove_spacing)
        if set_current:
            self.current_section = section_title

        return

    def delete_section_by_title(self, section_title: str):
        """
        Deletes a section of the report given a title string.

        Parameters
        ----------
        section_title : str
            Title of the section to delete.
        """
        if section_title not in self.sections.keys():
            # Do not raise an error, just continue
            return
        del self.sections[section_title]
        if self.current_section == section_title:
            if len(self.sections) > 0:
                self.current_section = next(reversed(self.sections)).title()
            else:
                self.current_section = None

    def write(self, content: str, section_title: str | None = None) -> None:
        """
        Write string content to a given section.

        Parameters
        ----------
        content : str
            Text to write to the section.
        section_title : str | None, default=None
            If provided, write to a given section. If None is provided, write to the current section.
        """
        if section_title is None:
            section_title = self.current_section
        section = self.get_section(section_title) #ensure section exists
        section.add_content(content)

        
        
    def generate(self) -> str:
        """
        Creates the combined markdown string from all of the sections.

        Returns
        -------
        output : str
           Combined markdown string.
        """
        output = ""
        for key in self.sections.keys(): # check the ordering
            section = self.sections[key]
            output += section.generate()
        return output

    

    def generate_pdf(self, filename: str, verbose: bool = False) -> None:
        """
        Creates a PDF from the combined markdown string.

        Parameters
        ----------
        filename : str
           Output filename for the PDF.
        verbose : bool, default=False
           If True, prints the command generated
        """
        command_list = [
                "pandoc",
                "--from",
                "markdown",
                "--to",
                "pdf",
                "--metadata",
                f"title={self.title}",
                "--quiet",
                "-o",
                filename,
                "--pdf-engine",
                "weasyprint",
            ]
        if verbose:
            print(" ".join(command_list))
            print(self.header)
        subprocess.run(command_list,
            text=True,
            input=self.header + self.generate(),
        )


def color_text(text: str, color: str = "black", highlight: str = "white") -> str:
    """
    Convenience function to generate colored text in a markdown string via HTML

    Parameters
    ----------
    text : str
       Text to color.
    color : str, default="black"
       Color of the text.
    highlight : str, default="white"
       Color of the highlight/background.
    """
    return f"<span style=\"color:{color}; background-color: {highlight};\">{text}</span>"


