#!/usr/bin/env bash

pyenv_installed="$(brew list pyenv | grep pyenv)"
if [[ $"pyenv_installed" == "" ]]; then
  echo "... pyenv 설치"
  brew install pyenv
fi

pipenv_installed="$(brew list pipenv | grep pipenv)"
if [[ $"pipenv_installed" == "" ]]; then
  echo "... pipenv 설치"
  brew install pipenv
fi

pyenv local 3.9.13
pipenv install

# nodejs 설치
pipenv run nodeenv --node 18.17.1 --python-virtualenv

