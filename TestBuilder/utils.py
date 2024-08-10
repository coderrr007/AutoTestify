import os
import git
from git import Repo
from urllib.parse import quote
from .testcase_helpers import setup_openai, create_testcases


def clone_repo(repo_link, clone_dir, username=None, password=None):
    try:
        if username and password:
            username = quote(username)
            password = quote(password)
            repo_link = repo_link.replace("https://", f"https://{username}:{password}@")
        if not os.path.exists(clone_dir):
            os.makedirs(clone_dir)
        Repo.clone_from(repo_link, clone_dir)
        return True, "Repository cloned successfully."
    except git.exc.GitCommandError as e:
        return False, str(e)


def execute_testcase_generation(
    openai_key,
    openai_type,
    openai_version,
    openai_base_url,
    project_path,
    neo4j_url,
    username,
    password,
):
    setup_openai(openai_key, openai_type, openai_base_url, openai_version)
    create_testcases(openai_type, project_path, neo4j_url, username, password)
