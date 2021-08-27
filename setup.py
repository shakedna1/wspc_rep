import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="wspc",
    version="0.0.6",
    author="Shaked Naor-Hoffmann & ‪Dina Svetlitsky",
    author_email="zivukelson@gmail.com",
    description="Protein-content-based bacterial pathogenicity classifier",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/shakedna1/wspc_rep",
    project_urls={
        "Bug Tracker": "https://github.com/shakedna1/wspc_rep/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    install_requires=[
        'pandas>=1.2.4',
        'numpy>=1.20.3',
        'scikit-learn==0.24.2',
        'scipy>=1.6.3'
    ],
    include_package_data=True,
    package_data={
        "wspc": ["src/wspc/model/WSPC_model.pkl"],
    },
    entry_points={
        'console_scripts': [
            'wspc = wspc:command_line.main',
        ],
    },
)
