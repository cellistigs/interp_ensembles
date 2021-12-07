import setuptools
import os

loc = os.path.abspath(os.path.dirname(__file__))

#with open(os.path.join(loc,"../","README.md"),"r") as fh:
#    long_description = fh.read()

setuptools.setup(
        name = "interpensembles",
        version = "0.0.1",
        author = "Taiga Abe",
        author_email = "ta2507@columbia.edu",
        description = "Interpolation-ensembles", 
        long_description = "",
        long_description_content_type = "test/markdown", 
        url = "https://github.com/cellistigs/interp-ensembles",
        packages = setuptools.find_packages(),
        include_package_data=True,
        package_data={},
        classifiers = [
            "License :: OSI Approved :: MIT License"],
        python_requires=">=3.6",
        )

