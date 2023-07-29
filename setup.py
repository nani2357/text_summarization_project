# Import the setuptools module, necessary for packaging Python projects
import setuptools

# Open and read the contents of the README.md file
with open("README.md", "r", encoding ="utf-8") as f:
    long_description = f.read()
    
    # Define the version of the package
    __version__ = "0.0.0"
    
    # Define the repository name, author's username, source repository, and author's email
    REPO_NAME = "text_summarization_project"
    AUTHOR_USER_NAME = "nani2357"
    SRC_REPO = "textSummarization"
    AUTHOR_EMAIL = "naveenkadampally@gmail.com"
    
    # Use setuptools to package the project
    setuptools.setup(
        # The name of the package
        name=SRC_REPO,
        # The version of the package
        version=__version__,
        # The author's username
        author=AUTHOR_USER_NAME,
        # The author's email
        author_email=AUTHOR_EMAIL,
        # A short description of the project
        description="NLP_text_summarization_project",
        # A long description of the project, typically read from the README
        long_description=long_description,
        # The format of the long description
        long_description_content_type="text/markdown",
        # The URL of the project's main homepage
        url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
        # Additional URLs for the project
        project_urls={
            "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
        },
        # A dictionary mapping package source directories to package names
        package_dir={"": "src"},
        # Automatically discover all packages in the specified directory
        packages=setuptools.find_packages(where="src")
    )
