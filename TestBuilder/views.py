from django.shortcuts import render, redirect
from django.http import HttpResponse
from .forms import RepoForm, OpenAIForm
from django.conf import settings
from .utils import clone_repo, execute_testcase_generation

def repo_form_view(request):
    if request.method == 'POST':
        form = RepoForm(request.POST)
        if form.is_valid():
            repo_link = form.cleaned_data['repo_link']
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']
            clone_dir = getattr(settings, 'CLONE_DIR', 'cloned_repos')
            success, message = clone_repo(repo_link, clone_dir, username=username, password=password)
            if success:
                return redirect('openai_form')
            else:
                form.add_error(None, message)
    else:
        form = RepoForm()
    return render(request, 'TestBuilder/repo_form.html', {'form': form})

def openai_form_view(request):
    if request.method == 'POST':
        form = OpenAIForm(request.POST)
        if form.is_valid():
            openai_key = form.cleaned_data['openai_key']
            openai_version = form.cleaned_data['openai_version']
            openai_type = form.cleaned_data['openai_type']
            openai_base_url = form.cleaned_data['openai_base_url']
            neo4j_url = form.cleaned_data['neo4j_url']
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']
            clone_dir = getattr(settings, 'CLONE_DIR', 'cloned_repos')
            execute_testcase_generation(openai_key, openai_type, openai_version, openai_base_url, clone_dir, neo4j_url, username, password)
            return HttpResponse("Success")
    else:
        form = OpenAIForm()
    return render(request, 'TestBuilder/openai_form.html', {'form': form})
