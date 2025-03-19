import os
import tempfile
import subprocess
from langchain.schema import Document


def git_loader(
    repo_url: str, branch: str = "main"
) -> tuple[list[Document], list[Document]]:
    with tempfile.TemporaryDirectory() as temp_dir:
        # Clone repository using subprocess
        subprocess.run(
            ["git", "clone", "--branch", branch, repo_url, temp_dir], check=True
        )

        py_docs = []
        md_docs = []

        # Walk through the directory and collect files
        for root, _, files in os.walk(temp_dir):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    if file.endswith(".py"):
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                            py_docs.append(
                                Document(
                                    page_content=content, metadata={"source": file_path}
                                )
                            )
                    elif file.endswith(".md"):
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                            md_docs.append(
                                Document(
                                    page_content=content, metadata={"source": file_path}
                                )
                            )
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

        return py_docs, md_docs
