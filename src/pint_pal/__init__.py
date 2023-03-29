import pint_pal.checkin
try:
    from setuptools_scm import get_version
    __version__ = get_version(root='..', relative_to=__file__)
except (LookupError, ModuleNotFoundError):
    # not an editable install, use build-time version
    from pint_pal._version import __version__
