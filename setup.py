import setuptools


with open("README.md", "r", encoding ="utf-8") as f:
    long_description = f.read()
    
    __version__ = "0.0.0"
    
    REPO_NAME = "text_summarization_project"
    AUTHOR_USER_NAME = "nani2357"
    SRC_REPO = "text_summarization"
    AUTHOR_EMAIL = "naveenkadampally@gmail.com"
    
    setuptools.setup(
        name=SRC_REPO,
        version=__version__,
        author=AUTHOR_USER_NAME,
        author_email=AUTHOR_EMAIL,
        description="NLP_text_summarization_project",
        long_description=long_description,
        long_description_content="text/markdown",
        url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
        project_urls={
            "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
        },
        package_dir={"": "src"},
        packages=setuptools.find_packages(where="src")
    )
