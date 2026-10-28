"""Testes da validação local de nomes no chat, sem chamada à API Gemini."""
from __future__ import annotations

import sys
import types
import unittest

# A validação é local; os stubs permitem executar os testes sem instalar Streamlit
# ou google-genai. Em uma instalação real, os pacotes existentes são preservados.
try:
    import streamlit  # noqa: F401
except ModuleNotFoundError:
    streamlit_stub = types.ModuleType("streamlit")
    streamlit_stub.secrets = {}
    streamlit_stub.cache_resource = lambda **kwargs: (lambda func: func)
    sys.modules["streamlit"] = streamlit_stub

try:
    from google import genai  # noqa: F401
    from google.genai import types as genai_types  # noqa: F401
except (ModuleNotFoundError, ImportError):
    google_stub = types.ModuleType("google")
    genai_stub = types.ModuleType("google.genai")
    genai_types_stub = types.ModuleType("google.genai.types")

    class ClientStub:
        def __init__(self, *args, **kwargs):
            pass

    genai_stub.Client = ClientStub
    genai_stub.types = genai_types_stub
    genai_types_stub.GenerateContentConfig = lambda **kwargs: kwargs
    google_stub.genai = genai_stub
    sys.modules["google"] = google_stub
    sys.modules["google.genai"] = genai_stub
    sys.modules["google.genai.types"] = genai_types_stub

import pandas as pd

import assistente_ia as assistente


def criar_base():
    df = pd.DataFrame({
        "Valor": [100.0, 200.0, 300.0, 400.0],
        "Cliente Gerencial": [
            "Mercado Beta", "Mercado Beto", "Rede Horizonte", "Cliente Exato"
        ],
        "Produto": [
            "Lavadora Ultra", "Bomba Pressurizadora", "Lavadora Compacta", "Mercado Bet"
        ],
        "Vendedor": ["Vendedor A", "Vendedor B", "Vendedor C", "Vendedor D"],
        "SKU": ["BP500", "BP600", "BP700", "BP800"],
    })
    cols = {
        "valor": "Valor", "qtd": None, "data": None,
        "cliente": "Cliente Gerencial", "produto": "Produto", "sku": "SKU",
        "vendedor": "Vendedor", "representante": None, "vendedor_cadastro": None,
        "gerente_regiao": None, "segmento": None, "empresa": None,
        "status_faturamento": None, "departamento": None,
        "_colunas_anonimizacao": {}, "_dinamicas": {},
        "_prefixos_sensiveis": {}, "_numericas": {},
    }
    return df, cols


class TestValidacaoPalavras(unittest.TestCase):
    def setUp(self):
        assistente._CACHE_CANDIDATOS_FUZZY.clear()
        self.df, self.cols = criar_base()
        self.quadro = assistente.Quadro(self.df, self.cols)

    def test_corrige_nome_com_erro_mesmo_com_outro_nome_exato_na_pergunta(self):
        pergunta = (
            "Mostre o faturamento do produto Lavadora Ultra e do cliente Mercdo Beta"
        )
        corrigida, correcoes, sugestao, meta = assistente.corrigir_termos_similares(
            pergunta, self.quadro
        )
        self.assertIn("cliente Mercado Beta", corrigida)
        self.assertEqual(correcoes, [("Mercdo Beta", "Mercado Beta")])
        self.assertIsNone(sugestao)
        self.assertIsNone(meta)

    def test_preserva_nome_exato_sem_sugerir_correcoes(self):
        pergunta = "Mostre o faturamento do cliente Mercado Beta"
        resultado = assistente.corrigir_termos_similares(pergunta, self.quadro)
        self.assertEqual(resultado[0], pergunta)
        self.assertEqual(resultado[1], [])
        self.assertIsNone(resultado[2])
        self.assertIsNone(resultado[3])

    def test_nao_tenta_corrigir_palavra_sem_semelhanca_suficiente(self):
        resultado = assistente.corrigir_termos_similares(
            "Explique uma tendência sem relação com nomes de clientes", self.quadro
        )
        self.assertEqual(resultado[0], "Explique uma tendência sem relação com nomes de clientes")
        self.assertEqual(resultado[1], [])
        self.assertIsNone(resultado[2])

    def test_candidatos_ambiguos_pedem_confirmacao(self):
        resultado = assistente.corrigir_termos_similares(
            "Mostre o faturamento do cliente Mercado Bet", self.quadro
        )
        self.assertEqual(resultado[1], [])
        self.assertIn("mais de um nome", resultado[2])
        self.assertTrue(resultado[3]["ambigua"])
        self.assertEqual(resultado[3]["opcoes"], ["Mercado Beta", "Mercado Beto"])

    def test_contexto_cliente_nao_se_confunde_com_produto_exato(self):
        df = pd.DataFrame({
            "Valor": [100.0],
            "Cliente Gerencial": ["Lavadora Ultro"],
            "Produto": ["Lavadora Ultra"],
        })
        cols = dict(self.cols)
        cols["sku"] = None
        q = assistente.Quadro(df, cols)
        resultado = assistente.corrigir_termos_similares(
            "Mostre o faturamento do cliente Lavadora Ultra", q
        )
        self.assertIn("cliente Lavadora Ultro", resultado[0])
        self.assertEqual(resultado[1], [("Lavadora Ultra", "Lavadora Ultro")])
        self.assertIsNone(resultado[2])

    def test_codigo_com_erro_pede_confirmacao_em_vez_de_trocar_silenciosamente(self):
        resultado = assistente.corrigir_termos_similares("Faturamento do SKU BP50O", self.quadro)
        self.assertIn("BP50O", resultado[0])
        self.assertIn("você quis dizer", resultado[2])
        self.assertEqual(resultado[3]["candidato"], "BP500")

    def test_cache_invalida_quando_valores_mudam_sem_mudar_formato(self):
        self.df.loc[0, "Cliente Gerencial"] = "Mercado Gama"
        q = assistente.Quadro(self.df, self.cols)
        resultado = assistente.corrigir_termos_similares(
            "Faturamento do cliente Mercdo Gama", q
        )
        self.assertEqual(resultado[1], [("Mercdo Gama", "Mercado Gama")])


    def test_confirmacao_sim_aplica_o_termo_correto_na_pergunta_original(self):
        pendente = {
            "pergunta_original": "Faturamento do cliente Mercdo Beta",
            "termo": "Mercdo Beta", "candidato": "Mercado Beta",
            "opcoes": ["Mercado Beta"], "ambigua": False,
        }
        resultado = assistente.resolver_confirmacao_validacao("Sim.", pendente)
        self.assertEqual(resultado["acao"], "confirmada")
        self.assertEqual(resultado["pergunta"], "Faturamento do cliente Mercado Beta")
        self.assertEqual(resultado["correcoes"], [("Mercdo Beta", "Mercado Beta")])

    def test_confirmacao_ambigua_aceita_opcao_numerada(self):
        pendente = {
            "pergunta_original": "Faturamento do cliente Mercado Bet",
            "termo": "Mercado Bet", "candidato": "Mercado Beta",
            "opcoes": ["Mercado Beta", "Mercado Beto"], "ambigua": True,
        }
        resultado = assistente.resolver_confirmacao_validacao("opção 2", pendente)
        self.assertEqual(resultado["acao"], "confirmada")
        self.assertEqual(resultado["pergunta"], "Faturamento do cliente Mercado Beto")

    def test_confirmacao_negativa_nao_consulta_a_sugestao(self):
        pendente = {
            "pergunta_original": "Faturamento do cliente Mercdo Beta",
            "termo": "Mercdo Beta", "candidato": "Mercado Beta",
            "opcoes": ["Mercado Beta"], "ambigua": False,
        }
        resultado = assistente.resolver_confirmacao_validacao("Não!", pendente)
        self.assertEqual(resultado["acao"], "recusada")
        self.assertIsNone(resultado["pergunta"])

    def test_processar_pergunta_devolve_confirmacao_para_o_chat(self):
        def gerar_nao_deve_ser_chamado(*args, **kwargs):
            self.fail("A validação local deve ocorrer antes de chamar o Gemini.")

        saida = assistente.processar_pergunta(
            self.df, self.cols, "Mostre o faturamento do cliente Mercado Bet",
            mapa={}, historico=[], gerar=gerar_nao_deve_ser_chamado,
        )
        self.assertEqual(saida["detalhes"]["rota"], "validacao_local_similaridade")
        self.assertEqual(
            saida["detalhes"]["opcoes_sugeridas"], ["Mercado Beta", "Mercado Beto"]
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
