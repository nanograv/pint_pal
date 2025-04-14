# test_report.py
# Jamison Talley
# 2024-06-20
from pint_pal.report import Report
import unittest
'''
Unit testing for report.py
'''



class ReportTests(unittest.TestCase):
    """ report.py testing class """

    def setUp(self):
        """ Load a TimingConfiguration object during setup """
        self.report = Report(title="Test Title")
        


    def test_new_section(self):
        """ Check adding new section """
        section_add = "Section 1"
        self.report.new_section(section_add)
        self.report.new_section(section_add)
        self.assertEqual(self.report.sections, ["Section 1"])


    def test_add_markdown(self):
        """ Check adding markdown to a report """
        content = ("# Testing Header" + "\n"+
                   "*Testing itallics*" + "\n"+
                   "**Testing bold**" + "\n"+
                   "Testing text" + "\n")
        self.report.add_markdown(section="Section 1",content=content,also_display=False)
        self.assertEqual(self.report.section_content["Section 1"].getvalue(), "\n\n" + content)


    def test_generate(self):
        """ Check the return of the generate """
        self.report = Report(title="Test Title")
        content = ("# Testing Header" + "\n"+
                   "*Testing itallics*" + "\n"+
                   "**Testing bold**" + "\n"+
                   "Testing text" + "\n")
        self.report.add_markdown(section="Section 1",content=content)
        result = ("# Test Title\n\n\n\n\n## Section 1\n\n\n\n\n# Testing Header\n*Testing itallics*\n**Testing bold**\nTesting text\n\n")
        self.assertEqual(self.report.generate(), result)


try:
    unittest.main(argv=[''], verbosity=2)
except SystemExit: #cleaner output below
    print()
