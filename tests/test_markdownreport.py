"""
Unit testing for markdownreport.py

Starting with generation using Claude, significant modification thereafter
"""

import pytest
from collections import OrderedDict
from unittest.mock import patch
from pint_pal.markdownreport import MarkdownSection, MarkdownReport, color_text


@pytest.fixture
def TestSection():
    title = "Test Section"
    content = ["Item 1", "Item 2"]
    section = MarkdownSection(title=title, content=content)
    return section


class TestMarkdownSection:
    """Test suite for MarkdownSection class"""

    def test_init(self, TestSection):
        """
        Test initialization with custom parameters
        """
        section = TestSection
        assert section.title == "Test Section"
        assert section.content == ["Item 1", "Item 2"]
        assert section.remove_spacing is False

    def test_property_getter_setter(self, TestSection):
        """Test getter and setter using title, content"""
        section = TestSection
        assert section.title == "Test Section"
        assert section.content == ["Item 1", "Item 2"]

        new_title = "New Title"
        new_content = ["Item 1", "Item 2", "Item 3"]
        section.title = "New Title"
        section.content = new_content
        assert section.title == new_title
        assert section.content == new_content

    def test_add_content(self):
        """Test adding content to section"""
        section = MarkdownSection()
        section.add_content("Item 1")
        section.add_content("Item 2")
        assert len(section.content) == 2
        assert section.content[0] == "Item 1"
        assert section.content[1] == "Item 2"

    def test_generate(self):
        """Test generate method with and without title included"""
        section = MarkdownSection(title="Test Section")
        section.add_content("Item 1")
        section.add_content("Item 2")
        output = section.generate(include_title=True)
        assert "## Test Section" in output
        assert "Item 1" in output
        assert "Item 2" in output

        output = section.generate(include_title=False)
        assert "## Test Section" not in output
        assert "Item 1" in output
        assert "Item 2" in output

    def test_generate_spacing(self, TestSection):
        """Test generate method with and without spacing set"""
        section = TestSection
        output = section.generate(include_title=False)
        assert "Item 1\n\n" in output
        assert "Item 2\n\n" in output

        section.remove_spacing = True
        output = section.generate(include_title=False)
        assert "Item 1    \n" in output
        assert "Item 2    \n" in output

    def test_generate_empty_content(self):
        """Test generate with no content"""
        section = MarkdownSection(title="Empty Section")
        output = section.generate(include_title=True)

        assert "## Empty Section" in output
        assert output.count("\n") >= 2  # Should have newlines from title


class TestMarkdownReport:
    """Test suite for MarkdownReport class"""

    def test_init_default(self):
        """Test initialization"""
        report = MarkdownReport(title="Test Report", font_size=12)
        assert report.title == "Test Report"
        assert isinstance(report.sections, OrderedDict)
        assert len(report.sections) == 0
        assert report.current_section is None
        assert "font-size: 12pt" in report.header

        sections = OrderedDict()
        sections["Test Section"] = MarkdownSection(title="Test Section")
        report = MarkdownReport(sections=sections)
        assert len(report.sections) == 1
        assert "Test Section" in report.sections

    def test_get_section_existing(self):
        """Test getting an existing section, remove_spacing works"""
        report = MarkdownReport()
        report.add_section_by_title("Section 1")
        section = report.get_section("Section 1")

        assert isinstance(section, MarkdownSection)
        assert section.title == "Section 1"
        assert section.remove_spacing is False

    def test_get_section_nonexistent(self):
        """Test getting a non-existent section raises KeyError"""
        report = MarkdownReport()
        with pytest.raises(KeyError, match="Section title not found: Nonexistent"):
            report.get_section("Nonexistent")

    def test_add_section_by_title(self):
        """Test adding a section by title, setting the current to True/False"""
        report = MarkdownReport()
        report.add_section_by_title("Section 1")

        assert "Section 1" in report.sections
        assert report.current_section == "Section 1"
        assert isinstance(report.sections["Section 1"], MarkdownSection)

        report.add_section_by_title("Section 2", set_current=False, remove_spacing=True)
        assert report.current_section == "Section 1"
        assert "Section 2" in report.sections


    def test_add_section_by_title_duplicate(self):
        """Test adding a duplicate section title raises KeyError"""
        report = MarkdownReport()
        report.add_section_by_title("Section 1")

        with pytest.raises(KeyError, match="Section title already exists: Section 1"):
            report.add_section_by_title("Section 1")

    def test_delete_section_by_title(self):
        """Test deleting an existing section"""
        report = MarkdownReport()
        report.add_section_by_title("Section 1")
        report.add_section_by_title("Section 2")

        report.delete_section_by_title("Section 1")

        assert "Section 1" not in report.sections
        assert "Section 2" in report.sections

    def test_delete_section_by_title_updates_current(self):
        """Test that deleting current section updates current_section"""
        report = MarkdownReport()
        report.add_section_by_title("Section 1")
        report.add_section_by_title("Section 2")

        report.delete_section_by_title("Section 2")

        assert report.current_section == "Section 1"

    def test_delete_last_section_sets_current_to_none(self):
        """Test that deleting the last section sets current_section to None"""
        report = MarkdownReport()
        report.add_section_by_title("Section 1")

        report.delete_section_by_title("Section 1")

        assert report.current_section is None
        assert len(report.sections) == 0

    def test_delete_nonexistent_section(self):
        """Test that deleting a non-existent section doesn't raise error"""
        report = MarkdownReport()
        # Should not raise an error
        report.delete_section_by_title("Nonexistent")

    def test_write_to_current_section(self):
        """Test writing to the current section"""
        report = MarkdownReport()
        report.add_section_by_title("Section 1")
        report.write("Item 1")

        section = report.get_section("Section 1")
        assert "Item 1" in section.content

    def test_write_to_specific_section(self):
        """Test writing to a specific section"""
        report = MarkdownReport()
        report.add_section_by_title("Section 1")
        report.add_section_by_title("Section 2")

        report.write("Item 1", section_title="Section 1")

        section1 = report.get_section("Section 1")
        section2 = report.get_section("Section 2")

        assert "Item 1" in section1.content
        assert len(section2.content) == 0

    def test_write_multiple_lines(self):
        """Test writing multiple lines to a section"""
        report = MarkdownReport()
        report.add_section_by_title("Test Section")

        report.write("Item 1")
        report.write("Item 2")
        report.write("Item 3")

        section = report.get_section("Test Section")
        assert len(section.content) == 3

    def test_generate_empty_report(self):
        """Test generating an empty report"""
        report = MarkdownReport()
        output = report.generate()
        assert output == ""

    def test_generate_sections(self):
        """Test generating a report with one section"""
        report = MarkdownReport()
        report.add_section_by_title("Section 1")
        report.write("Item 1")
        report.write("Item 2")
        report.add_section_by_title("Section 2")
        report.write("Item 3")
        report.add_section_by_title("Section 3")
        output = report.generate()
        assert "## Section 1" in output
        assert "Item 1" in output
        assert "Item 2" in output
        assert "## Section 2" in output
        assert "Item 3" in output

        # Test order preservation
        first_pos = output.find("## Section 1")
        second_pos = output.find("## Section 2")
        third_pos = output.find("## Section 3")

        assert first_pos < second_pos < third_pos

    @patch('subprocess.run')
    def test_generate_pdf_basic(self, mock_run):
        """Test PDF generation with default parameters"""
        report = MarkdownReport(title="Test Report")
        report.add_section_by_title("Section 1")
        report.write("Item 1")

        report.generate_pdf("output.pdf")

        mock_run.assert_called_once()
        call_args = mock_run.call_args
        command_list = call_args[0][0]

        # Check command list
        assert command_list[0] == "pandoc"
        assert "--to" in command_list
        assert "pdf" in command_list
        assert "output.pdf" in command_list
        assert "--pdf-engine" in command_list
        assert "weasyprint" in command_list

        # Check that input contains header and content
        assert "text" in call_args[1]
        assert call_args[1]["text"] is True
        assert "input" in call_args[1]
        assert "## Section 1" in call_args[1]["input"]

        # Check that title is in metadata
        title_index = command_list.index("--metadata")
        assert command_list[title_index + 1] == "title=Test Report"

    @patch('subprocess.run')
    def test_generate_pdf_with_verbose(self, mock_run, capsys):
        """Test PDF generation with verbose output"""
        report = MarkdownReport(title="Test Report")
        report.add_section_by_title("Section 1")

        report.generate_pdf("output.pdf", verbose=True)

        captured = capsys.readouterr()
        assert "pandoc" in captured.out
        assert "---" in captured.out  # Header should be printed



def test_color_text_defaults():
    """Test color_text with colors"""
    result = color_text("Warning", color="red", highlight="yellow")
    assert "Warning" in result
    assert "color:red" in result
    assert "background-color: yellow" in result
    assert 'style="' in result
    assert result.startswith("<span")
    assert result.endswith("</span>")


class TestIntegration:
    """Integration tests for the complete workflow"""

    def test_full_report_workflow(self):
        """Test creating a complete report from scratch"""
        report = MarkdownReport(title="Integration Test Report", font_size=11)

        # Add multiple sections
        report.add_section_by_title("Introduction")
        report.write("This is the introduction.")
        report.write("It has multiple paragraphs.")

        report.add_section_by_title("Methods")
        report.write("We used the following methods...")

        report.add_section_by_title("Results", remove_spacing=True)
        report.write("Result 1")
        report.write("Result 2")

        # Generate output
        output = report.generate()

        # Verify structure
        assert "## Introduction" in output
        assert "## Methods" in output
        assert "## Results" in output
        assert "This is the introduction." in output
        assert "We used the following methods..." in output
        assert "Result 1" in output

        # Verify ordering
        intro_pos = output.find("## Introduction")
        methods_pos = output.find("## Methods")
        results_pos = output.find("## Results")
        assert intro_pos < methods_pos < results_pos

    def test_section_manipulation_workflow(self):
        """Test adding, modifying, and deleting sections"""
        report = MarkdownReport()

        # Add sections
        report.add_section_by_title("Section 1")
        report.add_section_by_title("Section 2")
        report.add_section_by_title("Section 3")

        # Write to different sections
        report.write("Item 1", section_title="Section 1")
        report.write("Item 2", section_title="Section 2")
        report.write("Item 3")  # Current section

        # Delete a section
        report.delete_section_by_title("Section 2")

        # Verify final state
        output = report.generate()
        assert "## Section 1" in output
        assert "## Section 2" not in output
        assert "## Section 3" in output
        assert "Item 1" in output
        assert "Item 2" not in output
        assert "Item 3" in output

    def test_colored_text_in_report(self):
        """Test using colored text within a report"""
        report = MarkdownReport()
        report.add_section_by_title("Section 1")

        colored_content = color_text("Warning", color="red", highlight="yellow")
        report.write(colored_content)
        report.write("Normal text")

        output = report.generate()
        assert '<span style="color:red; background-color: yellow;">Warning</span>' in output
        assert "Normal text" in output
