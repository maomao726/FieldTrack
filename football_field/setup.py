import setuptools

print(setuptools.find_namespace_packages())

print("-----------")
print(setuptools.find_packages())

setuptools.setup(
    name="football_field",
    version="0.0.1",
    author="hcchen",
    author_email="l311551105.cs11@nycu.edu.tw",
    description="Football field detection",
    packages=setuptools.find_namespace_packages(),
    python_requires=">=3.6",
)