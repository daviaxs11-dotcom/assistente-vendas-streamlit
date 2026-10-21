@echo off
chcp 65001 >nul
title Painel de Vendas
cd /d "%~dp0"

if not exist "painel_vendas.py" (
    echo Nao encontrei o arquivo painel_vendas.py nesta pasta:
    echo %~dp0
    pause
    exit /b 1
)

rem --- 1. Ambiente virtual: cria se nao existir e ativa ---
if not exist ".venv\Scripts\activate.bat" (
    echo Ambiente virtual nao encontrado. Criando a pasta .venv ...
    python -m venv .venv
    if errorlevel 1 (
        echo.
        echo Nao foi possivel criar o ambiente virtual.
        echo Verifique se o Python esta instalado e se esta no PATH.
        pause
        exit /b 1
    )
)

call ".venv\Scripts\activate.bat"

rem --- 2. Pacotes: instala apenas se algum estiver faltando ---
python -c "import streamlit, pandas, plotly, openpyxl, google.genai" >nul 2>&1
if errorlevel 1 (
    echo Instalando os pacotes necessarios. Isso pode levar alguns minutos ...
    python -m pip install --upgrade pip
    if exist "requirements.txt" (
        python -m pip install -r requirements.txt
    ) else (
        python -m pip install streamlit pandas plotly openpyxl xlrd python-calamine google-genai
    )
    if errorlevel 1 (
        echo.
        echo Falha ao instalar os pacotes. Verifique a conexao com a internet.
        pause
        exit /b 1
    )
)

rem --- 3. Abre o painel ---
echo.
echo Abrindo o painel no navegador. Feche esta janela para encerrar o painel.
echo.
python -m streamlit run painel_vendas.py

echo.
echo O painel foi encerrado.
pause
