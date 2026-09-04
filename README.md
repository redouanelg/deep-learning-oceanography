# deep-learning-oceanography
Best practices in deep learning for oceanography

Some recommendations:

* Every example should do into a separate sub-directory.
* Put a README.md in your sub-directory
* Specify your dependency (`requirements.txt`/`pyproject.toml` for pip, `Project.toml` for julia,...)  so that the package manager can easily instantiate a project
* Share large datasets via a link and download them via a script
