import setuptools

setuptools.setup(
    name="catanatron_experimental",
    version="1.0.0",
    author="Bryan Collazo",
    author_email="bcollazo2010@gmail.com",
    description="Experimental scripts",
    url="https://github.com/bcollazo/catanatron",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "rich",
        "langchain-google-genai"
    ],  # careful including heavy ml-libs since might break heroku build
    entry_points={
        "console_scripts": ["catanatron-play=catanatron_experimental.play:simulate"],
    },
)
