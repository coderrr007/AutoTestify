# TestBuilder/forms.py
from django import forms

class RepoForm(forms.Form):
    repo_link = forms.URLField(label='GitHub Repository Link', required=True)
    username = forms.CharField(label='GitHub Username', required=True)
    password = forms.CharField(widget=forms.PasswordInput, required=True)

class OpenAIForm(forms.Form):
    openai_key = forms.CharField(label='OpenAI Key', required=True)
    openai_type = forms.CharField(label='OpenAI Type', required=True)
    openai_version = forms.CharField(label='OpenAI Version', required=True)
    openai_base_url = forms.URLField(label='OpenAI Base URL', required=True)
    neo4j_url = forms.CharField(label='Neo4j Url', required=True)
    username = forms.CharField(label='Neo4j Username', required=True)
    password = forms.CharField(widget=forms.PasswordInput, required=True)
